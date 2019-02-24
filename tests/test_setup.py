import pandas as pd
def get_data():
    df = pd.read_csv("./data/ames.csv", encoding="Latin-1", low_memory=False)
    return(df)