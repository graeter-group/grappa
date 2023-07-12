

## classes ##
class Atom:
    def __init__(self,idx,element):
        self.idx = idx
        self.element = element
        self.neighbors = []

    # neighbor is supposed to be a list of atoms
    def add_neighbors(self, neighbors: list):
        self.neighbors.extend(neighbors)

    def get_neighbor_elements(self,order):
        curr_neighbors = [self]
        for _ in range(order):
            prev_neighbors = curr_neighbors
            curr_neighbors = []
            for neighbor in prev_neighbors:
                curr_neighbors.extend(neighbor.neighbors)

        elements = ''.join(sorted([neighbor.element for neighbor in curr_neighbors]))
        return elements

    def get_neighbor_idxs(self,order):
        curr_neighbors = [self]
        for _ in range(order):
            prev_neighbors = curr_neighbors
            curr_neighbors = []
            for neighbor in prev_neighbors:
                curr_neighbors.extend(neighbor.neighbors)

        return [x.idx for x in curr_neighbors]

class AtomList:
    def __init__(self,atoms,bonds):
        self.idxs: list = []
        self.elements = []
        self.atoms: list[Atom] = []
        self.bonds = []
        self.bond_indices = bonds

        for atom in atoms:
            self.atoms.append(Atom(*atom))

        self.set_indices()
        self.set_elements()

        for bond in bonds:
            self.atoms[self.idxs.index(bond[0])].add_neighbors([self.atoms[self.idxs.index(bond[1])]])
            self.atoms[self.idxs.index(bond[1])].add_neighbors([self.atoms[self.idxs.index(bond[0])]])

        self.set_bonds()

    def set_indices(self):
        self.idxs: list = []
        for atom in self.atoms:
            self.idxs.append(atom.idx)

    def set_elements(self):
        self.elements = []
        for atom in self.atoms:
            self.elements.append(atom.element)

    def set_bonds(self):
        self.bonds: list = []
        for atom in self.atoms:
            for neighbor in atom.neighbors:
                self.bonds.append([atom,neighbor])


    def get_neighbor_elements(self,order):
        elements = []
        for atom in self.atoms:
            elements.append(atom.get_neighbor_elements(order))
        return elements

    def by_idx(self,idx):
        return self.atoms[self.idxs.index(idx)]
        
    def __len__(self):
        return len(self.atoms)
    
    def __repr__(self):
        return f"PDBData.AtomList object with {len(self.atoms)} atoms and {len(self.bonds)} bonds"