from pepgen.dataset import generate
from pathlib import Path


generate(n_max=50, length=1, outpath=str(Path(__file__).parent/'data/pep1'), choose_from=["A","G"], exclude=["J", "O"])