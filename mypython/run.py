import dataRead as dr
import dataExplore as de
import dataClean as dc
import modelBuild as mb
import modelVisualize as mv

import numpy as np
import pandas as pd

def main():
    print("Hello World!")
    #dr = dataRead(fullFilePath="../data/data.json")
    #dr.printDF()
    df = dr.readJSON("data/data.json")
    #de.checkDuplicatesInColumn(df, 'customer_id')
    de.checkDuplicates(df)
    de.checkMissingValues(df)
    de.getColumnNames(df)
    de.exploreData(df)
    de.getDataTypes(df)
    dc.encodeToNumeric(df, 'is_newsletter_subscriber')
    df = dc.removeColumnsWithMissingValues(df, ['customer_id', 'coupon_discount_applied'])
    cols = df.columns.values
    de.getOutliersForColumns(df)
    ocols = de.getColumnsForOutliers(df)
    z = dc.calcZscore(df[ocols])
    dfo = dc.removeOutliersUsingZscore(df[ocols], z)
    #print(len(dfo))
    #print(len(np.unique(np.where(z>3)[0])))
    rownumsToRemove = np.unique(np.where(z>3)[0])
    diffcols = list(set(cols).symmetric_difference(ocols))
    dfd = df[diffcols].drop(df.index[rownumsToRemove])
    #print(len(dfd))
    df1 = pd.concat([dfo, dfd], axis=1)
    #print(len(df1))
    #print(df1.columns.values)
    #print(df1.index)
    y_pred = mb.doKMeansDECClusters(df1)
    df2 = df1
    df2['gender'] = pd.Series(y_pred, index=df2.index)
    confusion_matrix, feature_importance = mb.doRFClassification(df2, 'gender')
    mv.visualizeHeatMapOfConfusionMatrix(confusion_matrix)
    mv.visualizeFeatureImportances(feature_importance)

if __name__ == "__main__":
    main()


