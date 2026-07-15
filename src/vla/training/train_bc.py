"""Training loop for the Phase-1 behavior-cloning baseline.

Standard supervised regression: predict the next joint-target from the current
observation, MSE on normalised actions, AdamW, mixed precision. Kept deliberately
plain — the baseline's value is as a correct, reproducible reference point, so
the loop favours legibility over cleverness. The checkpoint bundles the model
weights, its config, and the normalization stats so inference is self-contained.

Run with:
    python -m vla.training.train_bc --config configs/train_bc.yaml
or via the console script ``vla-train`` / ``scripts/train.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vla.data.dataset import (
    BCDataset,
    BCDatasetConfig,
    bc_collate,
    fit_stats,
    split_episodes,
)
from vla.data.lerobot_adapter import build_episode_index, load_lerobot_dataset
from vla.models.bc_policy import BCPolicy, BCPolicyConfig
from vla.utils.config import load_yaml, deep_update, parse_overrides
from vla.utils.logging import ExperimentLogger
from vla.utils.seed import set_seed


def build_datasets(cfg: dict):
    ds_cfg = cfg["dataset"]
    lerobot_ds = load_lerobot_dataset(ds_cfg["repo_id"], root=ds_cfg.get("root"))
    index = build_episode_index(lerobot_ds)

    bc_cfg = BCDatasetConfig(
        image_keys=ds_cfg["image_keys"],
        state_key=ds_cfg["state_key"],
        action_key=ds_cfg["action_key"],
        task_key=ds_cfg["task_key"],
        image_size=cfg["model"]["image_size"],
        val_fraction=ds_cfg["val_fraction"],
        seed=cfg["seed"],
    )
    train_eps, val_eps = split_episodes(index.num_episodes, bc_cfg.val_fraction, cfg["seed"])
    stats = fit_stats(lerobot_ds, bc_cfg, train_eps)

    train_ds = BCDataset(lerobot_ds, bc_cfg, train_eps, stats, train=True)
    val_ds = BCDataset(lerobot_ds, bc_cfg, val_eps, stats, train=False)
    return train_ds, val_ds, stats, bc_cfg


def evaluate_val(model: BCPolicy, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            pred = model(images, state, batch["instruction"])
            total += torch.nn.functional.mse_loss(pred, action, reduction="sum").item()
            n += action.numel()
    return total / max(n, 1)


def save_checkpoint(path: Path, model: BCPolicy, stats, cfg: dict, bc_cfg: BCDatasetConfig):
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "policy_config": asdict(model.cfg),
            "stats": {"state": asdict(stats.state), "action": asdict(stats.action)},
            "image_keys": bc_cfg.image_keys,
        },
        path,
    )


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = cfg["train"]

    train_ds, val_ds, stats, bc_cfg = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], collate_fn=bc_collate, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], collate_fn=bc_collate,
    )

    mcfg = cfg["model"]
    policy_cfg = BCPolicyConfig(
        clip_model_name=mcfg["clip_model_name"],
        n_cameras=mcfg["n_cameras"],
        state_dim=mcfg["state_dim"],
        action_dim=mcfg["action_dim"],
        hidden_dims=tuple(mcfg["hidden_dims"]),
        dropout=mcfg["dropout"],
    )
    model = BCPolicy(policy_cfg, stats).to(device)
    print(f"[train_bc] trainable params: {model.num_trainable_params():,} on {device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and device == "cuda")

    lcfg = cfg["logging"]
    logger = ExperimentLogger(lcfg["backend"], lcfg["project"], lcfg["run_name"], cfg)
    ckpt_dir = Path(lcfg["ckpt_dir"])

    step = 0
    best_val = float("inf")
    for epoch in range(tcfg["epochs"]):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=tcfg["amp"] and device == "cuda"):
                pred = model(images, state, batch["instruction"])
                loss = torch.nn.functional.mse_loss(pred, action)

            scaler.scale(loss).backward()
            if tcfg.get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            if step % lcfg["log_every"] == 0:
                logger.log({"train/loss": loss.item(), "epoch": epoch}, step)
            step += 1

        val_mse = evaluate_val(model, val_loader, device)
        logger.log({"val/mse": val_mse}, step)
        print(f"[train_bc] epoch {epoch}: val MSE = {val_mse:.5f}")

        if (epoch + 1) % lcfg["save_every_epochs"] == 0 or val_mse < best_val:
            best_val = min(best_val, val_mse)
            save_checkpoint(ckpt_dir / f"bc_epoch{epoch}.pt", model, stats, cfg, bc_cfg)
            save_checkpoint(ckpt_dir / "bc_best.pt", model, stats, cfg, bc_cfg)

    logger.finish()
    print(f"[train_bc] done. Best val MSE = {best_val:.5f}. Checkpoints in {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BC baseline policy.")
    parser.add_argument("--config", default="configs/train_bc.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Dotted overrides, e.g. train.lr=5e-4 seed=7")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.override:
        cfg = deep_update(cfg, parse_overrides(args.override))
    train(cfg)


if __name__ == "__main__":
    main()
