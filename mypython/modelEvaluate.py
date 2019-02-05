import numpy as np

def getConfusionMatrix(classifier, X_test, y_test):
    #print(classifier.feature_importances_)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
