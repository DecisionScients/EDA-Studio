#%%
from tests.test_setup import get_data
from eda_studio import describe
# Get Data
df = get_data()
s, d = describe.info(df)
#%%
print(d)