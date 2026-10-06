"""Build, save, and load FOM snapshot matrices used to train ROMs.

Convention used throughout burgers_rom: the snapshot matrix U is (n, m) --
n cells (rows) by m time instances (columns), so column j is the full spatial
solution at time t_j. This is the layout the POD/SVD and the DEIM flux
snapshot matrix expect directly, with no transposes at the call sites.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Snapshots:
    """FOM snapshot data, as saved by FOM/burgers1d/run_example.py.

    x: cell centers, shape (n,)
    t: sampled time instances, shape (m,)
    U: snapshot matrix, shape (n, m) -- one snapshot per column
    """

    x: np.ndarray
    t: np.ndarray
    U: np.ndarray


def load_snapshots(path: Path) -> Snapshots:
    """Load a .npz written by FOM/burgers1d/run_example.py.

    The FOM stores U time-major, (m, n); transpose once here so everything
    downstream sees the (n, m) column-per-snapshot convention.
    """
    data = np.load(path)
    return Snapshots(x=data["x"], t=data["t"], U=data["U"].T)
