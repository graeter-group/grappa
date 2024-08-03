## DGL Installation

For the application, a cpu installation of torch and dgl is usually sufficient.

To train grappa on a GPU, however, a CUDA version of DGL and torch needs to be installed along with as explained in https://www.dgl.ai/pages/start.html, e.g. by running
```
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html pyyaml pydantic
pip install torch==2.1.0
```

### A common DGL installation bug

A common installation problem of dgl (as of 2024) is that it installs incompatible torch versions and thus cannot find torch-version-named .so files:
```
python -c "import dgl"

-> FileNotFoundError: Cannot find DGL C++ graphbolt library at YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.3.1.so
```

To fix this, find a compatible torch version (in the example, 2.1.0):

```
ls YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt*

-> YOUR_ENVDIR/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.1.0.so
```

and install the torch version by hand:
```
pip install torch==2.1.0
```

Alternatively, one can try to use conda:

```
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/th21_cu118 dgl
```
