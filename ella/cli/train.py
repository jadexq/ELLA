import json
import shutil
import time
from datetime import timedelta
from pathlib import Path
from pprint import pp
from typing import Dict, List

import hydra
import lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from ella.cli.estimate import estimate
from ella.data.data_modules import SimulatedDataModule
from ella.models.cox import COX
from ella.options import ExperimentConfig, KernelParam

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def train(cfg: ExperimentConfig, kernel_idx: int, kernel_param: KernelParam) -> Path:
    cfg_dict: Dict = {
        "kernel_idx": kernel_idx,
        "kernel_param": kernel_param.model_dump(),
        **cfg.model_dump(exclude={"kernel_params", "estimation"}),
    }
    pp(cfg_dict)

    logger = TensorBoardLogger(
        save_dir=cfg.log.save_dir,
        name=None,
        version=f"gene_{cfg.data.gene_idx}-kernel_{kernel_idx}",
    )
    if Path(logger.log_dir).exists():
        if cfg.log.should_overwrite:
            shutil.rmtree(logger.log_dir)
        else:
            raise RuntimeError(f"{logger.log_dir} already exists")
    recipe_path = Path(logger.log_dir, "recipe.json")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    with recipe_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)

    dm = SimulatedDataModule(cfg.data, log_dir=logger.log_dir)
    model = COX(
        cfg=cfg.model,
        init_values=dm.model_init_values,
        kernel_idx=kernel_idx,
        kernel_param=kernel_param,
    )

    # Define EarlyStopping callback for reward
    early_stopping = EarlyStopping(
        monitor="train_loss",    # Monitor the `train_reward` or `train_loss`
        patience=10,             # Number of epochs to wait without improvement
        mode="min",              # `max` or `min` stop when reward stops increasing or loss stop decreasing
        verbose=True             # Enable logging for stopping messages
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", # `train_reward` or `train_loss`
        save_top_k=1,
        mode="min",           # `max` or `min` 
        filename="best-checkpoint",
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        callbacks=[early_stopping, checkpoint_callback],  # Add EarlyStopping callback here
    )
    trainer.fit(
        model=model,
        datamodule=dm,
    )

    return Path(logger.log_dir)


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(_cfg: DictConfig) -> None:
    start_time = time.time()

    cfg = ExperimentConfig(**hydra.utils.instantiate(_cfg))
    pp(cfg.model_dump())
    log_dirs: List[Path] = []

    cfg_kernel_null = cfg.copy(deep=True)
    cfg_kernel_null.model.beta_init = 0.0
    cfg_kernel_null.model.beta_requires_grad = False
    log_dir: Path = train(cfg=cfg_kernel_null, kernel_idx=-1, kernel_param=KernelParam(a0=1.0, b0=1.0))
    log_dirs.append(log_dir)

    for kernel_idx, kernel_param in enumerate(cfg.kernel_params):
        log_dir = train(cfg=cfg, kernel_idx=kernel_idx, kernel_param=kernel_param)
        log_dirs.append(log_dir)

    estimate(cfg=cfg.estimation, log_dirs=log_dirs)

    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"elapsed time = {elapsed_time}")
