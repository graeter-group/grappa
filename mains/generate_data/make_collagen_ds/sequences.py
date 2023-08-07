from grappa.constants import ONELETTER


def dipeptides_hyp_dop():
    AAs = [a for a in ONELETTER.values() if not a in ["B", "Z"]]
    out = []
    for AA1 in ["J", "O"]:
        for AA2 in AAs:
            out.append(AA1+AA2)
            out.append(AA2+AA1)
    return list(set(out))

for s in dipeptides_hyp_dop():
    print(s)