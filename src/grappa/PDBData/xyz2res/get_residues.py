DEBUG = False
#%%
import dgl
import torch
import json
import os.path
from grappa.units import RESIDUES
from pathlib import Path
#%%

def get_residue(subgraph, hashed_residues):

    elems = torch.argmax(subgraph.ndata["atomic_number"], dim=-1)
    # just take the square because it works as opposed to the sum
    res_hash = (elems*elems).sum().item()
    if not str(res_hash) in hashed_residues.keys():
        raise RuntimeError(f"Found a subgraph of length {len(elems)} that does not match any residue. Elements are: \n{elems}\nThe model predicts that the subgraph contains {subgraph.ndata['c_alpha'].sum().item()} c alpha atoms.\nConnectivity is: {subgraph.edges()},\nc alphas at: {subgraph.ndata['c_alpha']}, hash would be {res_hash}")
    res = hashed_residues[str(res_hash)]

    # these have the same elements, differentiate further:
    if res in ["ILE", "LEU"]:
        res = classify_leu(subgraph)
    return res

# only to be called if it is safe that the residue is either leu or ile
def classify_leu(subgraph):
    ca = torch.argwhere(subgraph.ndata["c_alpha"]!=0)[:,0].int().tolist()

    assert len(ca) == 1 , f"found invalid number of c alphas. c alphas are: {ca}"
    ca = ca[0]
    # get the carbon atom that comes next in the residue
    for m in subgraph.predecessors(ca):
        neighbor_elems = torch.argmax(subgraph.ndata["atomic_number"], dim=-1)[subgraph.predecessors(m).long()].tolist()


        # filter out the C by demanding that it shall not be connected to an O (sufficient for leu/ile)
        if not 8 in neighbor_elems and torch.argmax(subgraph.ndata["atomic_number"], dim=-1)[m].item() == 6:
            # we are at the C now. if this C has 2 C- neighbors, it is leucin, if it has 3 it is isoleucin
            n_c_neighbors = torch.where(torch.tensor(neighbor_elems)==6, 1,0).sum().item()
            if n_c_neighbors == 2:
                return "LEU"
            elif n_c_neighbors == 3:
                return "ILE"
            else:
                raise RuntimeError(f"Called function classify_leu on a graph that is neither leucin nor isoleucin: The C_beta has neither 2 nor 3 C-neighbors but {len(subgraph.predecessors(m))}")
    # if we have not returned yet, raise an exception
    raise RuntimeError("Called function classify_leu on a graph that is neither leucin nor isoleucin")


#%%
# get_residue: function that maps graph to residue string
def write_residues(g, get_residue=get_residue):

    if write_residues.hashed_residues is None:
        hash_storage = str(Path(__file__).parent/Path("scripts/hashed_residues.json"))
        if os.path.exists(hash_storage):
            with open(hash_storage, "r") as f:
                write_residues.hashed_residues = json.load(f)

    res_counter = 0

    elems = g.ndata["atomic_number"]
    elems = torch.argmax(elems, dim=-1)

    g.ndata["pred_residue"] = torch.ones(g.num_nodes())*-1
    g.ndata["res_number"] = torch.ones(g.num_nodes())*-1

    # returns an ordered list of c alphas. the ordering is starting at the C terminus and ending at the N terminus.
    c_alphas = get_ordered_c_alpha(g)

    # for each c alpha, cut the graph at the backbone
    # ie, on the backbone, find the next N ("N_remove") and the C that do not belong to this residue
    for n in c_alphas:
        neighbors = g.predecessors(n).tolist()
        N_remove = None
        C_remove = None
        CO = None
        for m in neighbors:
            # remove the C from the CO of the other residue
            if elems[m].item() == 7:
                # this N has only the C alpha, the to-remove C and an H as neighbor
                # in case or PRO, also a residue C is connected to it!
                # therefore check that the to-remove C has an O as neighbour
                for i in set(g.predecessors(m).tolist())-set({n}):
                    if elems[i].item() == 6:
                        if 8 in [elems[k] for k in g.predecessors(i)]:
                            C_remove = i
                            break

            # now search for the backbone N of the other residue
            # find the CO in our residue
            if elems[m].item() == 6:
                # we are at some C. if this is connected to an O, we remove the next N
                for i in g.predecessors(m).tolist():
                    if elems[i].item() == 8:
                        # the O must be connected to the C only.
                        if g.predecessors(i).tolist() == [m]:
                            # we are at the C from OC
                            CO = m
                            break

        if CO is None:
            raise RuntimeError("connectivity or the predicted c alphas are wrong, the backbone structure cannot be found")
        for i in g.predecessors(CO).tolist():
            if elems[i].item() == 7:
                # we are at the target N
                N_remove = i
                break

        if N_remove is None:
            raise RuntimeError("connectivity or the predicted c alphas are wrong, the backbone structure cannot be found")
        
        if C_remove is None:
            raise RuntimeError("connectivity or the predicted c alphas are wrong, the backbone structure cannot be found")

        # store node ids in g
        cutted_g = g
        cutted_g.ndata["old_node_id"] = torch.arange(g.num_nodes(), dtype=torch.int32)
        
        # take the subgraph containing the residue
        cutted_g = dgl.remove_nodes(cutted_g, [N_remove, C_remove])
        
        assert not n in [N_remove, C_remove]

        n_in_cutted = torch.argwhere(cutted_g.ndata["old_node_id"]==n)[0,0].item()
        
        res_nodes = find_connected_nodes(cutted_g, start_node=n_in_cutted)
        res_subgraph = dgl.node_subgraph(cutted_g, res_nodes)

        # predict residue
        try:
            residue = get_residue(res_subgraph, hashed_residues=write_residues.hashed_residues)
        except:
            if not DEBUG:
                raise
            else:
                dgl.save_graphs(str(Path(__file__).parent/Path("err_graphs.bin")),[g, res_subgraph])
                raise

        # translate to g indices
        res_nodes = cutted_g.ndata["old_node_id"][res_nodes].type(torch.int64)

        # write this into the graph
        res_idx = RESIDUES.index(residue)
        g.ndata["pred_residue"][res_nodes] = torch.ones(len(res_nodes))*res_idx

        # write residue number in the graph
        g.ndata["res_number"][res_nodes] = torch.ones(len(res_nodes))*res_counter

        res_counter += 1

    # identify the caps:
    if any(g.ndata["pred_residue"]<0):
        for i in range(2):
            cap_nodes = [n.item() for n in torch.argwhere(g.ndata["pred_residue"]<0).type(torch.int32)]

            # in case there is only one cap
            if len(cap_nodes) == 0:
                continue

            # store node ids in g
            all_cap_graph = g
            all_cap_graph.ndata["old_node_id"] = torch.arange(g.num_nodes(), dtype=torch.int32)

            # take the subgraph of unassigned residues
            all_cap_graph = dgl.node_subgraph(all_cap_graph, cap_nodes)

            # list of node indices in the subgraph, always start at zero and remove unassigned iteratively
            cap = find_connected_nodes(all_cap_graph, start_node=0)

            # translate to g indices
            cap = all_cap_graph.ndata["old_node_id"][cap].type(torch.int32)

            try:
                residue = get_residue(dgl.node_subgraph(g, cap), hashed_residues=write_residues.hashed_residues)
            except:
                if not DEBUG:
                    raise
                else:
                    dgl.save_graphs(str(Path(__file__).parent/Path("err_graphs.bin")),[g, dgl.node_subgraph(g, cap)])
                    raise

            res_idx = RESIDUES.index(residue)

            # write residue index in the graph
            g.ndata["pred_residue"][cap.long()] = torch.ones(len(cap))*res_idx

            if residue == "NME":
                res_number = torch.max(g.ndata["res_number"]).item()+1
            elif residue == "ACE":
                res_number = 0
                g.ndata["res_number"] += 1
            else:
                raise RuntimeError(f"Encountered unallowed (predicted) cap residue {residue}")

            # write residue number in the graph
            g.ndata["res_number"][cap.long()] = torch.ones(len(cap))*res_number

    return g
write_residues.hashed_residues = None
#%%

# returns an ordered list of c alphas. the ordering is starting at the C terminus and ending at the N terminus.
def get_ordered_c_alpha(g):
    c_alphas = torch.argwhere(g.ndata["c_alpha"]!=0)[:,0].tolist()
    if len(c_alphas) <= 1:
        return c_alphas

    def get_elem(g, n):
        return torch.argmax(g.ndata["atomic_number"], dim=-1)[n].item()

    # if to_left, we move towards the N terminus
    # called recursively until nothing is found anymore
    def append_c_alphas(start_node, ca=[], to_left=True):
        next_ca = None
        # move through the backbone respecting the order of the backbone atoms (to_left True or False)
        for m in g.predecessors(start_node):
            elem = get_elem(g, m)
            if (elem == 7 and to_left) or (elem == 6 and not to_left):

                for i in g.predecessors(m):
                    elem = get_elem(g, i)
                    if (elem == 6 and to_left) or (elem == 7 and not to_left):

                        for c in [entry.item() for entry in g.predecessors(i)]:
                            if c in c_alphas and not c in ca:
                                next_ca = c
                                break

                    if not next_ca is None:
                        break

            if not next_ca is None:
                break
        # recursion condition
        if next_ca is None:
            return ca
        else:
            ca.append(next_ca)
            # recursion step
            return append_c_alphas(start_node=next_ca, ca=ca, to_left=to_left)
    
    # find a sequential ordering in the graph
    # take some start c alpha and go left and right
    ca = append_c_alphas(c_alphas[0], [], True)
    ca.append(c_alphas[0])
    ca = append_c_alphas(c_alphas[0], ca=ca, to_left=False)

    if len(set(ca)) != len(ca):
        raise RuntimeError("found duplicates in ordered c_alpha list.")

    return ca


#%%
# returns the indices of all nodes that are connected to the start_node
def find_connected_nodes(g, start_node=0):
    nodeset = set()

    def process(node):
        nodeset.add(node)
        for neighbor in set(g.predecessors(node).tolist()) - nodeset:
            process(neighbor)

    process(start_node)
    return list(nodeset)
#%%
#%%
