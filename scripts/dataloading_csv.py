#%%
from grappa.data import Dataset
from grappa.utils.data_utils import get_moldata_path
import logging
logging.basicConfig(level=logging.INFO)
#%%
p = get_moldata_path('spice-dipeptide')
print(str(p))

#%%
# Load the dataset
# ds = Dataset.from_tag('spice-dipeptide')
# %%
# model loading:
from grappa.utils import model_from_tag

model = model_from_tag('grappa-1.3')


# model2 = model_from_tag('test_')