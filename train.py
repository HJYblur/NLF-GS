import argparse
import logging
import sys
import os
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
    parser.add_argument("--config", type=str, default="configs/nlfgs_base.yaml")
    args = parser.parse_args()
    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)
    logger = setup_logger(debug=False)
    logger.info("\n\n")
    logger.info("Starting training script")
    logger.info(f"Loaded config: {args.config}")

    # Determine device from config (fallback to cpu)
    device_str = None
    if isinstance(cfg, dict):
        device_str = cfg.get("sys", {}).get("device") if cfg.get("sys") else None
    # Fallback to environment variable or cpu
    if not device_str:
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device from config: {device}")

    # Prefer Tensor Cores / TF32 on Ampere+ GPUs for faster float32 matmul
    # See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    try:
        # Allow override from config: sys.matmul_precision: 'high' | 'medium' | 'highest'
        matmul_prec = None
        if isinstance(cfg, dict):
            matmul_prec = cfg.get("sys", {}).get("matmul_precision")
        if device.type == "cuda":
            torch.set_float32_matmul_precision(matmul_prec or "high")
            # Log the actual applied value
            logger.info(
                f"torch.set_float32_matmul_precision -> {torch.get_float32_matmul_precision()}"
            )
    except Exception as e:
        # Non-fatal; continue training with default behavior
        logger.warning(f"Could not set float32 matmul precision: {e}")

    # Build datamodule
    dm = AvatarDataModule(cfg)
    dm.setup("fit")
    logger.info("DataModule setup complete")

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

    # Log a small sanity-check about model param device/dtype (works with ScriptModule state_dict)
    try:
        sd = nlf_checkpoint.state_dict()
        first_tensor = next(iter(sd.values()))
        logger.debug(
            f"NLF checkpoint params -> device={first_tensor.device}, dtype={first_tensor.dtype}"
        )
    except Exception as _e:
        logger.debug(
            "Could not introspect NLF checkpoint state_dict for device/dtype check"
        )

    # Backbone Adapter Initialization
    backbone_cfg = cfg.get("backbone", {})
    use_resnet_fpn = bool(backbone_cfg.get("use_resnet_fpn", True))
    fpn_levels = tuple(backbone_cfg.get("fpn_levels", ["p2", "p3", "p4"]))
    backbone = FeatureExtractor(
        nlf_checkpoint,
        use_resnet_fpn=use_resnet_fpn,
        fpn_levels=fpn_levels,
        resnet_weights_path=backbone_cfg.get("resnet50_weights_path"),
    )
    logger.info("Backbone Adapter initialized")

    if use_resnet_fpn:
        fpn_out_channels = int(backbone_cfg.get("fpn_out_channels", 256))
        c_local = fpn_out_channels * len(fpn_levels)
    else:
        c_local = int(cfg["nlf"].get("latent_dim", 512))
    logger.info(f"Local feature dim: {c_local}")

    # Identity Encoder Initialization
    id_latent_dim = int(cfg["identity_encoder"].get("latent_dim", 64))
    id_encoder = IdentityEncoder(backbone_feat_dim=c_local, latent_dim=id_latent_dim)

    # Decoder Initialization
    decoder = GaussianDecoder()
    logger.info("Decoder initialized")

    # Renderer Initialization
    renderer = GsplatRenderer() if device != torch.device("cpu") else None
    logger.info("Renderer Initialized.")

    module = NlfGaussianModel(
        backbone_adapter=backbone,
        identity_encoder=id_encoder,
        decoder=decoder,
        renderer=renderer,
        train_decoder_only=True,
    )

    max_epochs = int(cfg["train"]["epochs"]) if "train" in cfg else 1
    logger.info(f"Training for max_epochs={max_epochs}")

    wandb_logger = WandbLogger(
        project="avatar-training",
        entity="lemon-tu-delft",
        save_dir=str(Path(__file__).parent / "logs"),
        log_model=False,
    )
    wandb_logger.log_hyperparams({"config": cfg, "config_path": args.config})

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
        max_epochs=max_epochs,
        devices=1,
        accelerator=cfg.get("train", {}).get("accelerator", "cpu"),
        precision=precision if precision else None,
        accumulate_grad_batches=accumulate,
        callbacks=callbacks if callbacks else None,
        logger=wandb_logger,
        log_every_n_steps=10,
    )
    logger.info("Beginning trainer.fit()")
    trainer.fit(module, datamodule=dm)
    logger.info("trainer.fit() finished")


def setup_logger(debug: bool = False) -> logging.Logger:
    """Initialize and return the project logger.

    If debug is True, set level to DEBUG; otherwise INFO. Writes logs to
    ./logs/train.log.
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "train.log"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
        ],
    )
    # Suppress noisy third-party DEBUG logs (e.g., PIL PNG plugin, fsspec)
    for name in (
        "PIL",
        "PIL.Image",
        "PIL.PngImagePlugin",
        "fsspec",
        "fsspec.local",
    ):
        try:
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False
        except Exception:
            pass
    return logging.getLogger("train")


if __name__ == "__main__":
    main()
