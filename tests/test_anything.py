#%%
import numpy as np
import pandas as pd
from tests.test_setup import get_data

# Get Data
df = get_data()
len(pd.isnull(df).any(1).to_numpy().nonzero()[0])

