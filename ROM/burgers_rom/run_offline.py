"""Offline stage: load FOM snapshots, compute the POD basis, plot the singular-value decay."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from burgers_rom.pod import compute_pod_basis
from burgers_rom.spod import compute_spod_basis
from burgers_rom.snapshots import load_snapshots

ROM = "sPOD" # POD or sPOD

repo_root = Path(__file__).parents[2]
snapshot_path = repo_root / "FOM" / "burgers1d" / "data" / "burgers1d_snapshots.npz"

snapshots = load_snapshots(snapshot_path)


if ROM == "POD":
    basis = compute_pod_basis(snapshots.U)
elif ROM == "sPOD":
    basis = compute_spod_basis(snapshots.U, snapshots.x)
else:
    raise ValueError(f"Unknown ROM type: {ROM}")


plotpath = Path(__file__).parent / "plots"
plotpath.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4.5))
mode_index = np.arange(1, basis.singular_values.size + 1)
ax.semilogy(mode_index, basis.normalized_singular_values(), marker="o", markersize=3)
ax.set_xlabel("mode index")
ax.set_ylabel(r"$\sigma_i / \sigma_0$")
ax.set_title(f"{ROM} singular-value decay - Burgers snapshots")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(plotpath / f"{ROM}_singular_values.png", dpi=150)
print(f"Saved {ROM}_singular_values.png")

for fraction in (0.90, 0.99, 0.999):
    r = basis.modes_for_energy(fraction)
    print(f"Modes needed for {fraction:.1%} energy: {r}")
