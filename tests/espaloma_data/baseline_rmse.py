import os, sys
import numpy as np
import glob
import random
import click
import espaloma as esp
from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers
import torch
# added for baseline force field calculation
from espaloma.data.md import *
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator
from openmm import unit, LangevinIntegrator
from openmm.app import Simulation
from openmm.unit import Quantity
import logging
logging.basicConfig(filename='logging.log', encoding='utf-8', level=logging.INFO)


# Parameters
HARTEE_TO_KCALPERMOL = 627.509
BOHR_TO_ANGSTROMS = 0.529177
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# Simulation Specs
TEMPERATURE = 350 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond
EPSILON_MIN = 0.05 * unit.kilojoules_per_mole


def load_data(input_prefix, dataset):
    """
    Load datasets
    """
    path = os.path.join(input_prefix, dataset)

    logging.info("Load unique molecules")
    ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
    logging.info("{} molecules found.".format(len(ds)))

    ds_tr, ds_vl, ds_te = None, None, None

    if dataset == "rna-nucleoside":
        ds_tr = ds
    elif dataset == "rna-trinucleotide":
        ds_te = ds
    else:
        ds_tr, ds_vl, ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
        logging.info(f"train:validate:test = {len(ds_tr)}:{len(ds_vl)}:{len(ds_te)}")

        # Load duplicated molecules
        logging.info("Load duplicated molecules")
        entries = glob.glob(os.path.join(input_prefix, "duplicated-isomeric-smiles-merge", "*"))
        random.seed(RANDOM_SEED)
        random.shuffle(entries)

        n_entries = len(entries)
        entries_tr = entries[:int(n_entries*TRAIN_RATIO)]
        entries_vl = entries[int(n_entries*TRAIN_RATIO):int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO)]
        entries_te = entries[int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO):]
        logging.info(f"Split duplicated dataset into {len(entries_tr)}:{len(entries_vl)}:{len(entries_te)} entries.")
        assert n_entries == len(entries_tr) + len(entries_vl) + len(entries_te)

        logging.info(f"Load only from {dataset}")
        for entry in entries_tr:
            _datasets = os.listdir(entry)
            for _dataset in _datasets:
                if _dataset == dataset:
                    _ds_tr = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                    ds_tr += _ds_tr
        for entry in entries_vl:
            _datasets = os.listdir(entry)
            for _dataset in _datasets:
                if _dataset == dataset:
                    _ds_vl = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                    ds_vl += _ds_vl
        for entry in entries_te:
            _datasets = os.listdir(entry)
            for _dataset in _datasets:
                if _dataset == dataset:
                    _ds_te = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                    ds_te += _ds_te
        
        logging.info(f"Final dataset split into {len(ds_tr)}:{len(ds_vl)}:{len(ds_te)}")

    return ds_tr, ds_vl, ds_te


def baseline_energy_force(g):
    """
    Compute baseline energy and force using openff-2.1.0 forcefield
    
    reference:
    https://github.com/choderalab/espaloma/espaloma/data/md.py
    """
    generator = SystemGenerator(
        small_molecule_forcefield="/home/takabak/.offxml/openff-2.1.0.offxml",
        molecules=[g.mol],
        forcefield_kwargs={"constraints": None, "removeCMMotion": False},
    )
    suffix = 'openff-2.1.0'

    # parameterize topology
    topology = g.mol.to_topology().to_openmm()
    # create openmm system
    system = generator.create_system(topology)
    # use langevin integrator, although it's not super useful here
    integrator = LangevinIntegrator(TEMPERATURE, COLLISION_RATE, STEP_SIZE)
    # create simulation
    simulation = Simulation(topology=topology, system=system, integrator=integrator)
    # get energy
    us = []
    us_prime = []
    xs = (
        Quantity(
            g.nodes["n1"].data["xyz"].detach().numpy(),
            esp.units.DISTANCE_UNIT,
        )
        .value_in_unit(unit.nanometer)
        .transpose((1, 0, 2))
    )
    for x in xs:
        simulation.context.setPositions(x)
        us.append(
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(esp.units.ENERGY_UNIT)
        )
        us_prime.append(
            simulation.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(esp.units.FORCE_UNIT) * -1
        )

    #us = torch.tensor(us)[None, :]
    us = torch.tensor(us, dtype=torch.float64)[None, :]
    us_prime = torch.tensor(
        np.stack(us_prime, axis=1),
        dtype=torch.get_default_dtype(),
    )

    g.nodes['g'].data['u_%s' % suffix] = us
    g.nodes['n1'].data['u_%s_prime' % suffix] = us_prime

    return g


def add_grad(g):
    g.nodes["n1"].data["xyz"].requires_grad = True
    return g


def _bootstrap_mol(x, y, n_samples=1000, ci=0.95):
    """
    """
    z = []
    for _x, _y in zip(x, y):
        mse = torch.nn.functional.mse_loss(_x, _y).item()
        z.append(np.sqrt(mse))
    z = np.array(z)
    mean = z.mean()

    results = []
    for _ in range(n_samples):
        _z = np.random.choice(z, z.size, replace=True)
        results.append(_z.mean())

    results = np.array(results)
    low = np.percentile(results, 100.0 * 0.5 * (1 - ci))
    high = np.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)

    return mean, low, high


def bootstrap_mol(u_ref, u, u_ref_prime, u_prime):
    """
    Bootstrap over molecules
    """
    mean, low, high = _bootstrap_mol(u_ref, u)
    ci_e = esp.metrics.latex_format_ci(
        mean * HARTEE_TO_KCALPERMOL, 
        low * HARTEE_TO_KCALPERMOL, 
        high * HARTEE_TO_KCALPERMOL
        )
    mean, low, high = _bootstrap_mol(u_ref_prime, u_prime)
    ci_f = esp.metrics.latex_format_ci(
        mean * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS), 
        low * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS), 
        high * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
        )

    return ci_e, ci_f


def inspect_rmse(ds, forcefields, suffix):
    """
    """
    # initialize
    mydict = {"u_qm": [], "u_qm_prime": []}
    for forcefield in forcefields:
        mydict["u_%s" % forcefield] = []
        mydict["u_%s_prime" % forcefield] = []
    # dataframe
    import pandas as pd
    df = pd.DataFrame(columns=["SMILES"] + [forcefield + "_ENERGY_RMSE" for forcefield in forcefields] + [forcefield + "_FORCE_RMSE" for forcefield in forcefields])
    # loop over molecule
    for g in ds:
        row = {}
        row["SMILES"] = g.mol.to_smiles()
        # center mean
        u_qm = (g.nodes['g'].data['u_qm'] - g.nodes['g'].data['u_qm'].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
        u_qm_prime = g.nodes['n1'].data['u_qm_prime'].detach().cpu().flatten()
        # append        
        mydict["u_qm"].append(u_qm)
        mydict["u_qm_prime"].append(u_qm_prime)
        for forcefield in forcefields:
            # center mean
            u = (g.nodes['g'].data['u_%s' % forcefield] - g.nodes['g'].data['u_%s' % forcefield].mean(dim=-1, keepdims=True)).detach().cpu().flatten()
            u_prime = g.nodes['n1'].data['u_%s_prime' % forcefield].detach().cpu().flatten()
            # rmse
            e_rmse = esp.metrics.rmse(u_qm, u) * HARTEE_TO_KCALPERMOL
            f_rmse = esp.metrics.rmse(u_qm_prime, u_prime) * (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS)
            # dataframe
            row[forcefield + "_ENERGY_RMSE"] = e_rmse.item()
            row[forcefield + "_FORCE_RMSE"] = f_rmse.item()
            # mydict
            mydict["u_%s" % forcefield].append(u)
            mydict["u_%s_prime" % forcefield].append(u_prime)
            print(forcefield, u_qm_prime.shape, u_prime.shape)
        df = df.append(row, ignore_index=True)
    # export dataframe
    if "openff-2.1.0" in forcefields:
        df = df.sort_values(by="openff-2.1.0_FORCE_RMSE", ascending=False) 
    else:
        df = df.sort_values(by="openff-2.0.0_FORCE_RMSE", ascending=False) 
    df.to_csv(f"inspect_{suffix}.csv")

    return mydict


def run(input_prefix, dataset, forcefields):
    """
    """
    def _fn(g):
        # remove
        g.nodes['n1'].data.pop('u_ref_prime')
        g.nodes['n1'].data.pop('q_ref')
        g.nodes['g'].data.pop('u_ref')
        g.nodes['g'].data.pop('u_ref_relative')
        # ensure precision match (saved with dtype fp64)
        g.nodes['g'].data['u_qm'] = g.nodes['g'].data['u_qm'].double()
        for forcefield in forcefields:
            g.nodes['g'].data['u_%s' % forcefield] = g.nodes['g'].data['u_%s' % forcefield].double()
        return g

    # load data
    ds_tr, ds_vl, ds_te = load_data(input_prefix, dataset)

    # compute rmse metric
    if ds_tr != None:
        ds_tr.apply(baseline_energy_force, in_place=True)
        ds_tr.apply(_fn, in_place=True)
        ds_tr.apply(add_grad, in_place=True)
        ds_tr.apply(regenerate_impropers, in_place=True)
        
        suffix = "tr"
        mydict = inspect_rmse(ds_tr, forcefields, suffix)
        with open(f"summary_{suffix}.csv", "w") as wf:
            wf.write("# energy / force\n")
            for forcefield in forcefields:
                ci_e, ci_f = bootstrap_mol(mydict["u_qm"], mydict["u_%s" % forcefield], mydict["u_qm_prime"], mydict["u_%s_prime" % forcefield])
                wf.write(f"{forcefield}: {ci_e} / {ci_f}\n")
    if ds_vl != None:
        ds_vl.apply(baseline_energy_force, in_place=True)
        ds_vl.apply(_fn, in_place=True)
        ds_vl.apply(add_grad, in_place=True)
        ds_vl.apply(regenerate_impropers, in_place=True)

        suffix = "vl"
        mydict = inspect_rmse(ds_vl, forcefields, suffix)
        with open(f"summary_{suffix}.csv", "w") as wf:
            wf.write("# energy / force\n")
            for forcefield in forcefields:
                ci_e, ci_f = bootstrap_mol(mydict["u_qm"], mydict["u_%s" % forcefield], mydict["u_qm_prime"], mydict["u_%s_prime" % forcefield])
                wf.write(f"{forcefield}: {ci_e} / {ci_f}\n")

    if ds_te != None:
        ds_te.apply(baseline_energy_force, in_place=True)
        ds_te.apply(_fn, in_place=True)
        ds_te.apply(add_grad, in_place=True)
        ds_te.apply(regenerate_impropers, in_place=True)

        suffix = "te"
        mydict = inspect_rmse(ds_te, forcefields, suffix)
        with open(f"summary_{suffix}.csv", "w") as wf:
            wf.write("# energy / force\n")
            for forcefield in forcefields:
                ci_e, ci_f = bootstrap_mol(mydict["u_qm"], mydict["u_%s" % forcefield], mydict["u_qm_prime"], mydict["u_%s_prime" % forcefield])
                wf.write(f"{forcefield}: {ci_e} / {ci_f}\n")


# @click.command()
# @click.option("-i", "--input_prefix", default="data", help="input prefix to graph data", type=str)
# @click.option("-d", "--dataset",      help="name of the dataset", type=str)
# @click.option("-f", "--forcefields",  help="baseline forcefields in sequence [gaff-1.81, gaff-2.10, openff-1.2.0, openff-2.0.0, amber14]", type=str)
# def cli(**kwargs):
#     print(kwargs)
#     input_prefix = kwargs['input_prefix']
#     dataset = kwargs['dataset']
#     _forcefields = kwargs['forcefields']
#     # convert forcefields into list
#     forcefields = [ str(_) for _ in _forcefields.split() ]

#     run(input_prefix, dataset, forcefields)


# if __name__ == '__main__':
#     cli()
                
from pathlib import Path

input_prefix = Path(__file__).parent.parent.parent/'data/esp_data'

dataset = 'gen2-torsion'

forcefields = ['openff-1.2.0']

run(input_prefix, dataset, forcefields)