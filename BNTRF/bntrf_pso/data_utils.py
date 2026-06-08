import math
import os
from typing import Tuple

import numpy as np


def read_tensor_file(
    file_path: str,
    separator: str = "::",
    log10_value_plus_one: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    indices = []
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(separator) if separator else line.split()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid format at line {line_no}: expected aID{separator}bID{separator}cID{separator}value"
                )
            aid = int(parts[0])
            bid = int(parts[1])
            cid = int(parts[2])
            value = float(parts[3])
            if log10_value_plus_one:
                value = math.log10(value + 1.0)
            indices.append((aid, bid, cid))
            values.append(value)

    if not indices:
        raise ValueError(f"Empty data file: {file_path}")

    return np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float64)


def get_index_ranges(train_indices: np.ndarray, valid_indices: np.ndarray, test_indices: np.ndarray):
    all_indices = np.vstack((train_indices, valid_indices, test_indices))
    max_aid = int(np.max(all_indices[:, 0]))
    max_bid = int(np.max(all_indices[:, 1]))
    max_cid = int(np.max(all_indices[:, 2]))
    min_aid = int(np.min(all_indices[:, 0]))
    min_bid = int(np.min(all_indices[:, 1]))
    min_cid = int(np.min(all_indices[:, 2]))
    return (min_aid, min_bid, min_cid), (max_aid, max_bid, max_cid)
