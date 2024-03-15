The json files store lists for every dataset:
[dsname, n_mols, n_confs, std_energies, std_forces, std_energies_std, std_forces_std, "forcefield": [rmse_energies-mean, rmse_energies-std, crmse_gradients-mean, crmse_gradients-mean]]
Units are kcal/mol and Angstrom. crmse is the componentwise-rmse, which is smaller by a factor of sqrt(3) than the actual force-vector rmse.