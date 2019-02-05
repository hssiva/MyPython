"""
 Description: Read an input file from the data folder
 Author: 'Siva'
"""
import json
import pandas as pd

def readJSON(filePath):
    with open(filePath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    #print(df.head())
    return df

def readCSV(filePath):
    with open(filePath) as f:
        df = pd.read_csv(f, sep=",")
    #print(df.head())
    return df

