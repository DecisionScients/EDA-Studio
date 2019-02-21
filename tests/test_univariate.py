#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), "tests"))
sys.path.append(os.path.join(os.getcwd(), "eda_studio"))

from test_setup import get_data
from univariate import Univariate
# Get Data
df = get_data()
analysis =  Univariate()

