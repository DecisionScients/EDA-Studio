#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), "tests"))
sys.path.append(os.path.join(os.getcwd(), "eda_studio"))

from test_setup import get_data
from univariate import Univariate
# Get Data
df = get_data()
analysis =  Univariate(df)
analysis.describe()
result = analysis.description
#%%
quant = result['quantitative']
qual = result['qualitative']
print(quant.info())
print(qual.info())
