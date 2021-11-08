import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def outliers_detection_choose(X):
    # detector_diget = input('\n(Only if in X there is no categorical features!! Initially absent or processed by encoding_models)\n'
    #                       'Choose the outliers detection method (enter a digit).\n'
    #                          '0: No outliers detection\n'
    #                          '1: Elliptic Envelope\n'
    #                          '2: One-Class SVM\n'
    #                          '3: Isolation Forest\n'
    #                          '4: LocalOutlierFactor\n')

    detector_diget = '3'

    if detector_diget == '0':
        return X

    if detector_diget == '1':
        clf = EllipticEnvelope(random_state=0).fit(X)
        is_anomalies = clf.predict(X)
        unique, counts = np.unique(is_anomalies, return_counts=True)
        X['anomalie'] = is_anomalies

    if detector_diget == '2':
        clf = OneClassSVM(kernel="rbf").fit(X)
        is_anomalies = clf.predict(X)
        unique, counts = np.unique(is_anomalies, return_counts=True)
        X['anomalie'] = is_anomalies

    if detector_diget == '3':
        clf = IsolationForest(max_samples=100, contamination=.1).fit(X)
        is_anomalies = clf.predict(X)
        unique, counts = np.unique(is_anomalies, return_counts=True)
        X['anomalie'] = is_anomalies

    if detector_diget == '4':
        clf = LocalOutlierFactor().fit(X)
        is_anomalies = clf.fit_predict(X)
        unique, counts = np.unique(is_anomalies, return_counts=True)
        X['anomalie'] = is_anomalies


    print("Count of anomalies in dataset: ", dict(zip(unique, counts)))
    print(X)

    return X
