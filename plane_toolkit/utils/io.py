import yaml
import pandas as pd

def read_yaml(path):
    with open(path, 'r') as f:
        res = yaml.safe_load(f)
        return res
    
def read_csv(path, header=None):
    df = pd.read_csv(path, header=header)
    res = df.values
    return res