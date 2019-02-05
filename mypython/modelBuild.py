import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

class ClusteringLayer(tf.keras.layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = tf.keras.layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, int(input_dim)), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def autoencoder1(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = tf.keras.layers.Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = tf.keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = tf.keras.layers.Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = tf.keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = tf.keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return tf.keras.models.Model(inputs=input_img, outputs=decoded, name='AE'), tf.keras.models.Model(inputs=input_img, outputs=encoded, name='encoder')

def doKMeansDECClusters(df, seed=289383, n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, n_jobs=4, random_state=seed)
    y_pred_kmeans = kmeans.fit_predict(df)
    pd.Series(y_pred_kmeans).value_counts()
    print(df.shape[-1])
    print(len(df))
    #return y_pred_kmeans

    np.random.seed(123)
    tf.set_random_seed(456)

    #Hyper-params
    dims = [df.shape[-1], 10]
    init = tf.keras.initializers.VarianceScaling(scale=1./5., mode='fan_in', distribution='uniform')
    pretrain_optimizer = tf.keras.optimizers.Adam() #SGD(lr=100,momentum=0.9) #.Adam()
    autoencoder, encoder = autoencoder1(dims, init=init)
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mean_squared_error') #loss='binary_crossentropy')
    autoencoder.fit(df, df, batch_size=100, epochs=30)
    autoencoder.save_weights('output/ae_weightsK2.h5')

    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = tf.keras.models.Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer='adam', loss='kld')
    y_pred = kmeans.fit_predict(encoder.predict(df))

    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    
    loss = 0
    index = 0
    maxiter = 45000
    update_interval = 1000
    index_array = np.arange(df.shape[0])
    tol = 0.001 # tolerance threshold to shop training
    
    y = y_pred_kmeans
    batch_size=1000
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(df, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                acc = np.round(metrics.accuracy_score(y, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, df.shape[0])]
        loss = model.train_on_batch(x=df.values[idx], y=p[idx.tolist()])
        index = index + 1 if (index + 1) * batch_size <= df.shape[0] else 0

    model.save_weights('output/DEC_model_final.h5')
    
    # Final Eval.
    q = model.predict(df.values, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc = np.round(metrics.accuracy_score(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        loss = np.round(loss, 5)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
    return y_pred

def doDTClassification(df, targetColName):
    X = df.drop(targetColName, axis=1)
    y = df[targetColName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    classifier = DecisionTreeClassifier()  
    classifier.fit(X_train, y_train)
    dotfile = open("dt.dot", 'w')
    export_graphviz(classifier, out_file=dotfile, feature_names=X.columns)
    dotfile.close()
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 

def doRFClassification(df, targetColName):
    X = df.drop(targetColName, axis=1)
    y = df[targetColName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    rfClassifier = RandomForestClassifier(n_estimators=100)
    rfClassifier.fit(X_train, y_train)
    y_predrf = rfClassifier.predict(X_test)
    print(confusion_matrix(y_test, y_predrf))  
    print(classification_report(y_test, y_predrf)) 
    feature_importance_rf = pd.Series(rfClassifier.feature_importances_,index=X_train.columns)
    print(feature_importance_rf)
    return confusion_matrix(y_test, y_predrf), feature_importance_rf

def doSVMClassification(df, targetColName):
    X = df.drop(targetColName, axis=1)
    y = df[targetColName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    svcClassifier = SVC(kernel='linear')
    svcClassifier.fit(X_train, y_train)
    y_predSVC = svcClassifier.predict(X_test)
    print(confusion_matrix(y_test,y_predSVC))  
    print(classification_report(y_test,y_predSVC))

def doXGBoostClassification(df, targetColName):
    X = df.drop(targetColName, axis=1)
    y = df[targetColName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    xgbClassifier = XGBClassifier()
    xgbClassifier.fit(X_train, y_train)
    y_predXGB = xgbClassifier.predict(X_test)
    print(confusion_matrix(y_test,y_predXGB))  
    print(classification_report(y_test,y_predXGB))
    print(X_train.columns)
    print(xgbClassifier.feature_importances_)
