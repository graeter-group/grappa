import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def generate_summary(targetpath, summarypath):
    targetpath = Path(targetpath)
    summarypath = Path(summarypath)
    summarypath.mkdir(exist_ok=True, parents=True)

    qm_energies = []
    gaff_energies = []
    qm_grads = []
    gaff_grads = []

    mols = 0
    confs = 0

    for npz_file in targetpath.glob("*.npz"):
        data = np.load(npz_file)

        qm_energies.append(data['u_qm'] - np.mean(data['u_qm']))
        gaff_energies.append(data['u_gaff-2.11'] - np.mean(data['u_gaff-2.11']))

        qm_grads.append(data['u_qm_prime'].flatten())
        gaff_grads.append(data['u_gaff-2.11_prime'].flatten())

        confs += data['xyz'].shape[0]
        mols += 1

    if len(qm_energies) == 0:
        raise ValueError(f"No .npz files found in {targetpath}")

    qm_energies = np.concatenate(qm_energies)
    gaff_energies = np.concatenate(gaff_energies)

    qm_grads = np.concatenate(qm_grads)
    gaff_grads = np.concatenate(gaff_grads)

    energy_rmse = np.sqrt(np.mean((qm_energies - gaff_energies)**2))
    grad_crmse = np.sqrt(np.mean((qm_grads - gaff_grads)**2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(qm_energies, gaff_energies, s=1, alpha=0.3)
    axes[0].set_xlabel('QM')
    axes[0].set_ylabel('GAFF-2.11')
    axes[0].text(0.05, 0.95, f"RMSE: {energy_rmse:.2f} kcal/mol", transform=axes[0].transAxes, verticalalignment='top')

    axes[1].scatter(qm_grads, gaff_grads, s=1, alpha=0.3)
    axes[1].set_xlabel('QM')
    axes[1].set_ylabel('GAFF-2.11')
    axes[1].text(0.05, 0.95, f"CRMSE: {grad_crmse:.2f} kcal/mol/Å", transform=axes[1].transAxes, verticalalignment='top')

    plt.savefig(str(summarypath / "summary.png"), dpi=300)

    summary = {
        "u_gaff_rmse": round(float(energy_rmse), 2),
        "grad_gaff_crmse": round(float(grad_crmse), 2),
        "total_mols": mols,
        "total_confs": confs,
    }

    with open(str(summarypath / "summary.json"), 'w') as file:
        json.dump(summary, file, indent=4)

    print(f"Summary generated at {summarypath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targetpath",
        type=str,
        default="/hits/fast/mbm/seutelf/data/datasets/spice-dipeptide",
        help="Path where .npz files are located.",
    )
    parser.add_argument(
        "--summarypath",
        type=str,
        default="/hits/fast/mbm/seutelf/espaloma_orig/summaries/spice-dipeptide",
        help="Path to store the summary."
    )
    args = parser.parse_args()
    generate_summary(targetpath=args.targetpath, summarypath=args.summarypath)
