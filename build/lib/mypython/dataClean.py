from scipy import stats
import numpy as np

### Manually convert columns to numeric as appropriate
def encodeToNumeric(df, colname):
    cleanup_nums = {colname: {"Y":1,"N":0}}
    df.replace(cleanup_nums, inplace=True)
    return df

def removeColumnsWithMissingValues(df, colNamesList):
    df = df.drop(colNamesList, axis=1)
    return df

def calcZscore(df):
    z = np.abs(stats.zscore(df))
    #print(z)
    #print(len(np.where(z>3)[0]))
    #print(len(np.where(z>4)[0]))
    return z

def removeOutliersUsingZscore(df,z):
    df = df[(z<3).all(axis=1)]
    return df
