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

from src.avatar_utils.config import load_config
from src.training.nlfgs_builder import (
    apply_matmul_precision_for_device,
    build_nlf_gaussian_model,
    device_from_cfg,
)
from src.data.datamodule import AvatarDataModule


def main():
    # Arg parsing
    parser = argparse.ArgumentParser(description="NLF-GS Training Scaffold")
    parser.add_argument("--config", type=str, default="configs/nlfgs_gpu.yaml")
    args = parser.parse_args()
    os.environ["NLFGS_CONFIG"] = args.config
    cfg = load_config(args.config)

    device = device_from_cfg(cfg)
    apply_matmul_precision_for_device(cfg, device)

    # Build datamodule
    dm = AvatarDataModule(cfg)
    dm.setup("fit")

    module = build_nlf_gaussian_model(cfg, device)

    max_epochs = int(cfg["train"]["epochs"]) if "train" in cfg else 1

    wandb_logger = WandbLogger(
        project="avatar-training",
        entity="lemon-tu-delft",
        log_model=False,
    )
    # Push full config to wandb run config (path + nested YAML tree + snapshots).
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
        from src.training.nlfgs_training_utils import GpuMemoryProfilerCallback
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
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
