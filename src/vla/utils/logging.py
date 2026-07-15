"""Thin experiment-logging wrapper over W&B / TensorBoard / no-op.

The training loop should not care which backend is configured, so it talks to
this uniform interface. Keeping it minimal (log scalars, log a config dict,
finish) avoids coupling the science code to a vendor SDK.
"""
from __future__ import annotations

from typing import Any


class ExperimentLogger:
    def __init__(self, backend: str, project: str, run_name: str, config: dict[str, Any]):
        self.backend = backend
        self._writer = None
        self._wandb = None

        if backend == "wandb":
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config)
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=f"runs/{run_name}")
        elif backend == "none":
            pass
        else:
            raise ValueError(f"Unknown logging backend: {backend}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)
        # "none": drop.

    def finish(self) -> None:
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.finish()
        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
