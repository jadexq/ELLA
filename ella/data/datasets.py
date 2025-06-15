import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SimulatedDataset(Dataset):
    def __init__(self, data_path: str, gene_idx: int, log_dir: str) -> None:
        self.cells: List[Dict] = []
        with Path(data_path).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if gene_idx == idx:
                    line_data: Dict = json.loads(line)
                    self.cells = line_data["cells"]
                    
                    gene_id: str = line_data["gene_id"]
                    dataset_info_path = Path(log_dir, "dataset_info.json")
                    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
                    with dataset_info_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "gene_id": gene_id,
                            },
                            f,
                            indent=4,
                        )
                    break
        assert len(self.cells) > 0, f"gene_idx {gene_idx} not found"
        self.model_init_values: Dict = {
            "lam_null": line_data["lam_null"],
        }
    
    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        cell = self.cells[idx]
        points = torch.tensor(cell["points"], dtype=torch.float32)
        sc_total = torch.tensor(cell["sc_total"], dtype=torch.float32)
        return points, sc_total
