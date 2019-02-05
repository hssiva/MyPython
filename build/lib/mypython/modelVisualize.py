import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd

def visualizeHeatMapOfConfusionMatrix(confusion_matrix):
    sns.set(font_scale=3)
    #confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()

def visualizeFeatureImportances(featureImportance):
    #(pd.Series(featureImportance, index=df.columns)
    #.nlargest(4)
    #.plot(kind='barh'))
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    # Load the data
    df = pd.DataFrame({'fimp': list(featureImportance), 'colName': list(featureImportance.index.values)})
    #print(df.columns)
    sns.factorplot("fimp","colName", data=df,kind="bar",palette="Blues",size=6,aspect=2,legend_out=False)

    plt.show()