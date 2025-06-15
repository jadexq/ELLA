from typing import List

from pydantic import BaseModel


class LogConfig(BaseModel):
    should_overwrite: bool = False
    save_dir: str = "lightning_logs"


class DataConfig(BaseModel):
    data_path: str
    gene_idx: int


class OptimizerConfig(BaseModel):
    lr: float


class KernelParam(BaseModel):
    a0: float
    b0: float


class ModelConfig(BaseModel):
    is_debug: bool
    optimizer: OptimizerConfig
    n_bins: int
    beta_init: float
    sigma02_init: float
    beta_requires_grad: bool = True


class TrainerConfig(BaseModel):
    max_epochs: int
    enable_progress_bar: bool = True


class EstimationConfig(BaseModel):
    search_dir: str
    log_dir_pattern: str
    output_path: str
    n_bins: int


class ExperimentConfig(BaseModel):
    log: LogConfig
    data: DataConfig
    kernel_params: List[KernelParam]
    model: ModelConfig
    trainer: TrainerConfig
    estimation: EstimationConfig
