#%%
from tests.test_setup import get_data
from eda_studio.univariate import Univariate
# Get Data
df = get_data()
analysis =  Univariate(df)
result = analysis.describe(df)
#%%
print(result['qualitative'])