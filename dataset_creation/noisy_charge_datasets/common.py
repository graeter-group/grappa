from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


GRAPPA_REPO = Path("/hits/basement/mli/seutelf/grappa")
GRAPPA_SRC = GRAPPA_REPO / "src"
DATASET_ROOT = GRAPPA_REPO / "data" / "datasets"

if str(GRAPPA_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPPA_SRC))


PUBCHEM_MANIFEST_PATH = Path(__file__).resolve().parent / "spice_pubchem_filtered_atom_counts.npz"


def format_std(std: float) -> str:
    return np.format_float_positional(float(std), trim="-")


def dataset_dir(tag: str) -> Path:
    return DATASET_ROOT / tag


def sorted_npz_paths(tag: str) -> list[Path]:
    paths = list(dataset_dir(tag).glob("*.npz"))
    return sorted(paths, key=lambda path: int(path.stem))


def make_output_tag(source_tag: str, std: float, seed: int) -> str:
    return f"{source_tag}-charge-noise-std{format_std(std)}-seed{seed}"


def make_output_dir(source_tag: str, std: float, seed: int) -> Path:
    return dataset_dir(make_output_tag(source_tag, std, seed))


def make_mean_free_noise(charges: np.ndarray, rng: np.random.RandomState, std: float) -> tuple[np.ndarray, np.ndarray]:
    charges64 = np.asarray(charges, dtype=np.float64)
    noise64 = rng.normal(0.0, std, size=len(charges64))
    noise64 -= noise64.mean()
    noisy_charges64 = charges64 + noise64
    return noise64.astype(np.float32), noisy_charges64.astype(np.float32)


def make_add_data(charges: np.ndarray, noise: np.ndarray, noisy_charges: np.ndarray, std: float) -> dict[str, np.ndarray]:
    return {
        "original_partial_charges": np.asarray(charges, dtype=np.float32),
        "charge_noise": np.asarray(noise, dtype=np.float32),
        "noisy_partial_charges": np.asarray(noisy_charges, dtype=np.float32),
    }


def clamp_slice(total: int, start: int | None, stop: int | None) -> tuple[int, int]:
    start = 0 if start is None else max(0, int(start))
    stop = total if stop is None else min(total, int(stop))
    stop = max(start, stop)
    return start, stop


def load_partial_charge_counts(paths: list[Path]) -> np.ndarray:
    counts = np.zeros(len(paths), dtype=np.int64)
    for i, path in enumerate(paths):
        with np.load(path) as data:
            counts[i] = int(data["partial_charges"].shape[0])
    return counts


def load_or_create_pubchem_counts(paths: list[Path]) -> np.ndarray:
    expected_stems = np.array([path.name for path in paths])
    if PUBCHEM_MANIFEST_PATH.exists():
        with np.load(PUBCHEM_MANIFEST_PATH) as manifest:
            stems = manifest["stems"]
            atom_counts = manifest["atom_counts"]
        if len(stems) == len(expected_stems) and np.array_equal(stems, expected_stems):
            return atom_counts.astype(np.int64, copy=False)

    atom_counts = load_partial_charge_counts(paths)
    tmp_path = PUBCHEM_MANIFEST_PATH.with_suffix(".tmp.npz")
    np.savez(tmp_path, stems=expected_stems, atom_counts=atom_counts)
    tmp_path.replace(PUBCHEM_MANIFEST_PATH)
    return atom_counts


def rng_for_slice(seed: int, atom_counts: np.ndarray, start: int) -> np.random.RandomState:
    rng = np.random.RandomState(seed)
    n_skipped_draws = int(atom_counts[:start].sum())
    if n_skipped_draws > 0:
        rng.normal(0.0, 1.0, size=n_skipped_draws)
    return rng
