import argparse
import sys
import os
import json
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.datamodule import AvatarDataModule
from src.encoder.feature_extractor import FeatureExtractor
from src.encoder.identity_encoder import IdentityEncoder
from src.decoder.gaussian_decoder import GaussianDecoder
from src.render.gaussian_renderer import GsplatRenderer
from src.training.trainer import NlfGaussianModel
from src.avatar_utils.config import load_config


def main():
    # Arg parsing
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_gpu.yaml")
    args = parser.parse_args()
    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)

    # Determine device from config (fallback to cpu)
    device_str = None
    if isinstance(cfg, dict):
        device_str = cfg.get("sys", {}).get("device") if cfg.get("sys") else None
    # Fallback to environment variable or cpu
    if not device_str:
        device_str = "cpu"
    device = torch.device(device_str)

    # Prefer Tensor Cores / TF32 on Ampere+ GPUs for faster float32 matmul
    # See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    try:
        # Allow override from config: sys.matmul_precision: 'high' | 'medium' | 'highest'
        matmul_prec = None
        if isinstance(cfg, dict):
            matmul_prec = cfg.get("sys", {}).get("matmul_precision")
        if device.type == "cuda":
            torch.set_float32_matmul_precision(matmul_prec or "high")
    except Exception:
        # Non-fatal; continue training with default behavior
        pass

    # Build datamodule
    dm = AvatarDataModule(cfg)
    dm.setup("fit")

    # Import nlf model
    # Load TorchScript model; force it to the chosen device
    nlf_checkpoint = torch.jit.load(
        cfg["nlf"]["checkpoint_path"], map_location=device
    ).eval()
    try:
        nlf_checkpoint.to(device)
    except Exception:
        # Some TorchScript modules may not implement .to(); that's okay
        raise RuntimeError("NLF model does not support .to() method.")

    # Small sanity-check about model param device/dtype (works with ScriptModule state_dict)
    try:
        sd = nlf_checkpoint.state_dict()
        first_tensor = next(iter(sd.values()))
        _ = (first_tensor.device, first_tensor.dtype)
    except Exception:
        pass

    # Backbone Adapter Initialization
    backbone_cfg = cfg.get("backbone", {})
    train_cfg = cfg.get("train", {})
    train_decoder_only = bool(train_cfg.get("train_decoder_only", True))
    use_resnet_fpn = bool(backbone_cfg.get("use_resnet_fpn", True))
    fpn_levels = tuple(backbone_cfg.get("fpn_levels", ["p2", "p3", "p4"]))
    backbone = FeatureExtractor(
        nlf_checkpoint,
        use_resnet_fpn=use_resnet_fpn,
        fpn_levels=fpn_levels,
        resnet_weights_path=backbone_cfg.get("resnet50_weights_path"),
        freeze_resnet_fpn=train_decoder_only,
    )

    if use_resnet_fpn:
        fpn_out_channels = int(backbone_cfg.get("fpn_out_channels", 256))
        c_local = fpn_out_channels * len(fpn_levels)
    else:
        c_local = int(cfg["nlf"].get("latent_dim", 512))

    # Identity Encoder Initialization
    id_latent_dim = int(cfg["identity_encoder"].get("latent_dim", 64))
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)

    # Decoder Initialization
    decoder = GaussianDecoder()

    # Renderer Initialization
    renderer = GsplatRenderer() if device != torch.device("cpu") else None

    module = NlfGaussianModel(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
        renderer=renderer,
        train_decoder_only=train_decoder_only,
    )

    max_steps = int(cfg.get("train", {}).get("max_steps", -1))
    if max_steps <= 0:
        raise ValueError("train.max_steps must be > 0 for step-based two-phase training.")

    wandb_logger = WandbLogger(
        project="avatar-training",
        entity="lemon-tu-delft",
        log_model=False,
    )
    # Keep Lightning hyperparam logging minimal and push the full config directly
    # to wandb run config, which is more reliable for nested structures.
    wandb_logger.log_hyperparams({"config_path": args.config})
    if hasattr(wandb_logger, "experiment") and wandb_logger.experiment is not None:
        safe_cfg = cfg if isinstance(cfg, dict) else {}
        # 1) canonical nested config tree
        wandb_logger.experiment.config.update(safe_cfg, allow_val_change=True)
        # 2) raw config path and YAML snapshot for exact reproducibility
        config_text = Path(args.config).read_text(encoding="utf-8")
        wandb_logger.experiment.config.update(
            {
                "config_path": args.config,
                "config_yaml": config_text,
                "config_json": json.dumps(safe_cfg, sort_keys=True),
            },
            allow_val_change=True,
        )

    # Trainer precision and accelerator
    precision = cfg.get("train", {}).get("precision")
    accumulate = int(cfg.get("train", {}).get("accumulate_grad_batches", 1))

    # Improve CUDA memory behavior unless user overrides
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # GPU memory profiling callback
    profile_gpu = cfg.get("train", {}).get("profile_gpu", False)
    callbacks = []
    if profile_gpu and device.type == "cuda":
        from src.training.gpu_profiler import GpuMemoryProfilerCallback
        callbacks.append(GpuMemoryProfilerCallback())

    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        devices=1,
        accelerator=cfg.get("train", {}).get("accelerator", "cpu"),
        precision=precision if precision else None,
        accumulate_grad_batches=accumulate,
        callbacks=callbacks if callbacks else None,
        logger=wandb_logger,
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
