#%%
import numpy as np
import pandas as pd
from tests.test_setup import get_data
from eda_studio.univariate import Univariate
# Get Data
df = get_data()
d=df.dtypes[df.dtypes!='object'].index.values
print(d)

def sk(y):
    sign = np.where(y<0, 'negative', 'positive')
    skewed = np.where(abs(y) > 1, 'high ' + str(sign) + ' skew',
                np.where(abs(y) >= 0.5, 'moderate ' + str(sign) + ' skew',
                'symmetric'))
    return(skewed)    
skew = pd.DataFrame({'skew':df[d].skew(axis=0)})
print(skew)
skew['skewed'] = np.where(skew < -1, 'high negative skew',
                    np.where(skew <= -0.5, 'moderate negative skew',
                    np.where(skew < 0.5, 'symmetric',
                    np.where(skew < 1, 'moderate positive skew',
                    'high positive skew'))))
print(skew)                    