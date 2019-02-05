import pandas as pd
import numpy as np

def checkDuplicates(df):
    dfduplen = len(df[df.duplicated()])
    if dfduplen>0:
        print("Dataset size = %s, Number of duplicates = %s" % (len(df), dfduplen))
    return dfduplen

def checkDuplicatesInColumn(df,colname):
    #print(any(df[colname].duplicated()))
    #print(len(df[colname].unique()))
    print(len(df[colname]))

def checkMissingValues(df):
    missingValueStatus = df.isnull().sum()
    #print(pd.Series(missingValueStatus).nonzero())
    return pd.Series(missingValueStatus).nonzero()

def getColumnNames(df):
    #print(df.columns)
    return df.columns.values

def exploreData(df):
    #print(df.describe())
    return df.describe()

def getDataTypes(df):
    #print(df.dtypes)
    return df.dtypes

def getOutliersForColumns(df):
    outliersp = []
    outliersn = []
    colWithOutliers = []
    for col in list(df.columns.values):
        if (len(df[col].value_counts())>30) and (df[col].dtypes in ['int64','float64']):
            op = len(df[df[col]>df[col].mean()+3*df[col].std()][col])
            outliersp.append(op)
            on = len(df[df[col]<df[col].mean()-3*df[col].std()][col])
            outliersn.append(on)
            if op>0 or on>0:
                colWithOutliers.append(col)
    #print(outliersp)
    #print(outliersn)
    #print(colWithOutliers)
    return colWithOutliers

def getColumnsForOutliers(df):
    cols = []
    for col in list(df.columns.values):
        if (len(df[col].value_counts())>30) and (df[col].dtypes in ['int64','float64']):
            cols.append(col)
    #print(cols)
    return cols
