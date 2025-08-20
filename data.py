import pandas as pd
def load_data ():
    data = pd.read_csv("processed_data.csv")
    return data