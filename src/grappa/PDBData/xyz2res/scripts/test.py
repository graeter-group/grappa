#%%
if __name__ == "__main__":
    from grappa.PDBData.xyz2res import get_residues
    import dgl

    graphs, _ = dgl.load_graphs("./../data/pep2.bin")

    for g in graphs[:10]:
        g = get_residues.write_residues(g)
        print(all(g.ndata["pred_residue"] == g.ndata["residue"]))
        print(g.ndata["res_number"])
    # %%
    for g in graphs:
        g = get_residues.write_residues(g)
        assert all(g.ndata["pred_residue"] == g.ndata["residue"])
        assert all([g.ndata["res_number"][i] <= g.ndata["res_number"][i+1] for i in range(g.num_nodes()-1)])
    # %%
