from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


DROP_KEYS = {
    "add_data_charge_noise_sigma",
    "add_data_preserve_total_charge",
}


def sanitize_file(path: Path) -> bool:
    with np.load(path) as data:
        files = list(data.files)
        if not any(key in data.files for key in DROP_KEYS):
            return False
        payload = {key: data[key] for key in files if key not in DROP_KEYS}

    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **payload)
    tmp_path.replace(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip scalar noisy-charge metadata that breaks MolData loading.")
    parser.add_argument("dataset_dirs", nargs="+", help="Dataset directories to sanitize.")
    args = parser.parse_args()

    for dataset_dir in args.dataset_dirs:
        dataset_path = Path(dataset_dir)
        paths = sorted(dataset_path.glob("*.npz"), key=lambda path: int(path.stem))
        changed = 0
        for path in tqdm(paths, desc=str(dataset_path.name)):
            changed += int(sanitize_file(path))
        print(f"{dataset_path}: sanitized {changed} / {len(paths)} files")


if __name__ == "__main__":
    main()
