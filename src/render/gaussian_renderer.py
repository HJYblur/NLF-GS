from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from gsplat import rasterization

from avatar_utils.camera import load_camera_mapping
from avatar_utils.config import get_config
from avatar_utils.view_config import VIEW_ORDER


class GsplatRenderer:
    def __init__(self):
        self.sh_degree = get_config().get("decoder", {}).get("sh_degree", 3)

    def render(
        self,
        gaussian_3d: torch.Tensor,
        gaussian_params: dict[str, torch.Tensor],
        view_name: Union[str, Sequence[str]],
        camera_model: str = "pinhole",  # “pinhole”, “ortho”, “fisheye”, and “ftheta”. Default is “pinhole”
        save_folder_path: str = None,
        save_prefix: str = "debug",
        render_mode: str = "RGB",
        backgrounds: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Render the Gaussian splat representation into 2D images.

        Args:
            gaussian_3d: Tensor of shape (N, 3) representing 3D Gaussian centers.
            gaussian_params: Dictionary containing Gaussian parameters such as scales, rotations, alphas, etc.
            view_name: A single view label (e.g. ``\"0\"`` or ``\"180\"``) or a list of labels.

        Returns:
            Rendered images as a tensor of shape (B, H, W, 3).
        """
        # Process sh because gsplat expects SH coefficients with shape […, N, K, 3],
        # where K is the number of SH coefficients.
        shs = gaussian_params["sh"]  # (N, K), K = (sh_degree + 1)^2 * 3
        N, K = shs.shape
        assert (
            K == (self.sh_degree + 1) ** 2 * 3
        ), f"We expected SH shape (N, {(self.sh_degree + 1) ** 2 * 3}), got {shs.shape}"
        colors = shs.view(N, -1, 3)  # (N, K//3, 3)

        width, height = get_config().get("data", {}).get("image_size", (1024, 1024))

        # gsplat kernels in this environment don't support fp16 for projection.
        # Force rasterization inputs to fp32 while keeping the rest of the training in AMP.
        means32 = gaussian_3d.float().contiguous()
        quats32 = gaussian_params["rotation"].float().contiguous()
        scales32 = gaussian_params["scales"].float().contiguous()
        alphas32 = gaussian_params["alpha"].float().contiguous()
        colors32 = colors.float().contiguous()

        # Load precomputed camera matrices (batched if a list is provided)
        viewmats, Ks = load_camera_mapping(view_name)  # (B, 4, 4), (B, 3, 3)
        viewmats = viewmats.to(gaussian_3d.device).contiguous()
        Ks = Ks.to(gaussian_3d.device).contiguous()

        raster_kw = dict(
            means=means32,
            quats=quats32,
            scales=scales32,
            opacities=alphas32,
            sh_degree=self.sh_degree,
            colors=colors32,
            viewmats=viewmats,
            Ks=Ks,
            camera_model=camera_model,
            width=width,
            height=height,
            render_mode=render_mode,
        )
        if backgrounds is not None:
            raster_kw["backgrounds"] = backgrounds.to(gaussian_3d.device)

        rendered_imgs, rendered_alphas, meta = rasterization(**raster_kw)
        # rendered_imgs: (B, H, W, 3)
        if save_folder_path is not None:
            from torchvision.io import write_png
            from torchvision.transforms.functional import convert_image_dtype

            # Ensure the output directory exists
            out_dir = Path(save_folder_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Normalize view_name to a list
            if isinstance(view_name, str):
                view_names_list = [view_name]
            else:
                view_names_list = list(view_name)

            # Save rendered image for each view
            for idx, vname in enumerate(view_names_list):
                out_file = out_dir / f"{save_prefix}_{vname}.png"
                sample_img = (
                    rendered_imgs[idx].permute(2, 0, 1).to(torch.device("cpu"))
                )  # (3, H, W)
                img_to_save = convert_image_dtype(sample_img.clamp(0, 1), dtype=torch.uint8)
                write_png(img_to_save, str(out_file))
        return rendered_imgs  # (B, H, W, 3) where B=len(view_name) if list

    @torch.inference_mode()
    def render_canonical_views(
        self,
        gaussian_3d: torch.Tensor,
        gaussian_params: dict[str, torch.Tensor],
        save_dir: Union[str, Path],
        view_names: Optional[Sequence[str]] = None,
        save_prefix: str = "train",
    ) -> torch.Tensor:
        """Inference helper: renders all canonical orbit views (same order as training)."""
        views = list(view_names) if view_names is not None else list(VIEW_ORDER)
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        return self.render(
            gaussian_3d,
            gaussian_params,
            view_name=views,
            save_folder_path=str(out),
            save_prefix=save_prefix,
        )
