import numpy as np
import dgl
import torch

def invert_permutation(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s



def get_oneletter(threeletter):
    return get_oneletter.d[threeletter]

get_oneletter.d = {
        "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V"
    }