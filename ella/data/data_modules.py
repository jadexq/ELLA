from pathlib import Path
from typing import Tuple, Dict

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ella.data.datasets import SimulatedDataset
from ella.options import DataConfig


class SimulatedDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig, log_dir: str) -> None:
        self.data_path: str = cfg.data_path
        self.gene_idx: int = cfg.gene_idx
        self.log_dir: str = log_dir
        self.train_dataset: SimulatedDataset
        super().__init__()

    @property
    def model_init_values(self) -> Dict:
        self.setup("fit")
        model_init_values = self.train_dataset.model_init_values
        print(f"{model_init_values=}")
        return model_init_values
    
    def prepare_data(self) -> None:
        if not Path(self.data_path).exists():
            raise RuntimeError(f"Data not found in {self.data_path}")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = SimulatedDataset(
                data_path=self.data_path,
                gene_idx=self.gene_idx,
                log_dir=self.log_dir,
            )

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points, sc_total = zip(*batch)
        points_lengths = torch.tensor([len(x) for x in points])
        padded_points = pad_sequence(list(points), batch_first=True)
        return padded_points, points_lengths, torch.tensor(sc_total)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=len(self.train_dataset),
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
