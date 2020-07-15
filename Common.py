# #############################################################################
# Command line arguments:
import time
import argparse
import collections
import os
import sys
import warnings
from collections import Counter
from random import randint

from keras import Sequential, Model, Input
from keras.layers import SimpleRNN, Dropout, Dense, Activation, GRU, LSTM, Concatenate, Conv1D, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from umap.umap_ import UMAP
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import LocallyLinearEmbedding

import numpy as np
import pandas as pd
import joblib

from itertools import chain

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, EditedNearestNeighbours, TomekLinks, NearMiss, ClusterCentroids
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, PredefinedSplit, ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from GI_Features import gini_features
from PI_Features import permutation_features

if not sys.warnoptions:  # this is dangerous and a quickfix for annoying LinearSVC not converging
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

rand = 42
from multiprocessing import cpu_count

threads = cpu_count() // 2  # hyperthreading/SMT is not worth it

# import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # CPU only
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)


# #############################################################################
# # Command line arguments
def getCommandArgs(def_dataset, def_clf, def_fsel, def_fnum, def_scale, def_ext, def_class, def_train, def_test, def_smote=True):
    parser = argparse.ArgumentParser(description='Various parameters can be specified')

    parser.add_argument("--dataset", default=def_dataset, type=str, help="dataset to use")
    parser.add_argument("--classifier", default=def_clf, type=str, help="classifier to use")
    parser.add_argument("--features", default=def_fsel, type=int, help="features to use e.g. 1 uses knn features")
    parser.add_argument("--fnum", default=def_fnum, type=int, help="num of features to use e.g. 21")
    parser.add_argument("--scale", default=def_scale, type=str, help="processor to use")
    parser.add_argument("--extract", default=def_ext, type=str, help="feature extraction method, e.g. LDA")
    parser.add_argument("--multiclass", default=def_class, type=int, help="For multiclass instead of binary")
    parser.add_argument("--trainsize", default=def_train, type=float, help="Default train size is 50% but choose higher")
    parser.add_argument("--testsize", default=def_test, type=float, help="Default test size is 50% but choose higher")
    parser.add_argument("--smote", default=def_smote, type=int, help="oversampling 1(yes) or 0(no)")

    args = parser.parse_args()
    specified_dataset = args.dataset
    specified_classifier = args.classifier
    specified_fsel = args.features
    specified_fnum = args.fnum
    specified_scaler = args.scale
    specified_ext = args.extract
    multiclass = args.multiclass
    train_size = args.trainsize
    test_size = args.testsize
    use_smote = args.smote
    return specified_dataset, specified_classifier, specified_fsel, specified_fnum, specified_scaler, specified_ext, multiclass, train_size, test_size, use_smote


# #############################################################################
# # Load data or model from a path:
def load_all_data(data_path, file):
    csv_path = data_path + file
    return pd.read_csv(csv_path)


# def fast_load_data(data_path, non_numeric=['Label', 'imbalance_label', 'is_malicious'], ex_class=[], dtype='object'):
def fast_load_data(data_path, non_numeric=['Label'], ex_class=[], dtype='object'):
    columns_to_skip = non_numeric
    df = pd.read_csv(data_path, engine='c', dtype=dtype, usecols=lambda x: x not in columns_to_skip)
    df2 = pd.read_csv(data_path, engine='c', dtype='object', usecols=lambda x: x in columns_to_skip)

    con = pd.concat([df, df2], axis=1)
    con = con[-con['Label'].isin(ex_class)] if ex_class != None else print()  # remove class(es) in Label
    return con


# #############################################################################
# # print samples per class:
def sample_cnt(y):
    (unique, counts) = np.unique(y, return_counts=True)
    return np.asarray((unique, counts)).T


# #############################################################################
# Encode train and test labels (necessary for certain classfiers like RF and neuralnets)
def encode_labels(y_train, y_test):
    le_train = preprocessing.LabelEncoder()
    le_train.fit(y_train.ravel())
    y_train_enc = le_train.transform(y_train.ravel())
    le_test = preprocessing.LabelEncoder()
    le_test.fit(y_test.ravel())
    y_test_enc = le_test.transform(y_test.ravel())
    return y_train_enc, y_test_enc


# #############################################################################
# Feature Selection
def pd_feature_sel(data, train_size, test_size, multiclass, stratify=None, orig_labels=False):
    # Separating features by storing binary and multiclass labels
    mc_labels = data.loc[:, ['Label']].values if orig_labels == True else data.loc[:, ['imbalance_label']].values
    features = data.iloc[:, :-1]  # -3 if binary labels included

    # bc_labels = data.loc[:, ['is_malicious']].values

    # all_features = features.columns
    # all_features = all_features[0:]
    # features = features.values

    # print(all_features)

    feature_selection = permutation_features()
    # feature_selection = gini_features()

    if multiclass == 1:  # multiclass split, first train/test and then further split train into train/val
        X_train, X_test, y_train, y_test = train_test_split(features, mc_labels, train_size=train_size, test_size=test_size, random_state=rand,
                                                            stratify=mc_labels)

        if stratify != None:
            print("Stratifying multiclass...")
            # X_train, y_train = initialSampling(X_train, y_train)
            # X_train, y_train = resample(X_train, y_train)
    else:
        if stratify != None:
            print("Stratifying binclass..")
            # stratify = bc_labels

        X_train, X_test, y_train, y_test = train_test_split(features, bc_labels, train_size=train_size, test_size=test_size, random_state=rand,
                                                            stratify=stratify)
    print("Train/Test Split Complete!")
    return X_train, X_test, y_train, y_test, feature_selection


# #############################################################################
# Initial sampling to ensure fairness and prevent severely undersampled classes from being excluded:
def initialSampling(X, y):
    min_sm = 5
    sampling_strat = dict(Counter((y.ravel())))
    print("Random (over) fix extremely underrepresented classes with ROS")
    sampling_strat = {key: min_sm - value + value + 1 if value <= min_sm else value for (key, value) in sampling_strat.items()}
    over = RandomOverSampler(random_state=rand, sampling_strategy=sampling_strat)
    X, y = over.fit_resample(X, y)
    return X, y


# #############################################################################
# Resample the dataset for balance
def resample(X, y, use_smote=0):
    if use_smote == 1:
        kn = 5
        X, y = initialSampling(X, y)  # duplication: make sure there are enough samples in extreme cases for smote
        print("SMOTE oversampling")
        sampling_strat = dict(Counter((y.ravel())))
        sampling_strat = {key: value + 195 if value <= 5000 else value for (key, value) in sampling_strat.items()}
        over = SMOTE(random_state=rand, k_neighbors=kn, sampling_strategy=sampling_strat, n_jobs=threads)
        X, y = over.fit_resample(X, y)

    sampling_strat = dict(Counter(y.ravel()))
    # print('Corrected sample distribution:\n {}'.format(sampling_strat))
    # Now undersample so that major classes are capped
    # print("Now Random (under) to cap majority classes")
    # biggest = 0
    # for key, value in sampling_strat.items():
    #     if biggest <= value:
    #         biggest = value
    # red_factor = 3
    # sampling_strat = {key: int(value / red_factor) if value >= int(biggest * 0.5) else value for (key, value) in sampling_strat.items()}
    # sampling_strat = {key: int(value / 1.2) if value >= int(biggest * (0.5/(red_factor*red_factor))) else value for (key, value) in sampling_strat.items()}
    max_samples = 50000
    # max_samples = 75000
    sampling_strat = {key: int((max_samples / value) * value) if value >= max_samples else value for (key, value) in sampling_strat.items()}
    under = RandomUnderSampler(random_state=rand, sampling_strategy=sampling_strat)
    X, y = under.fit_resample(X, y)

    # NCR combines the Condensed Nearest Neighbor (CNN) Rule to remove redundant examples and the Edited Nearest Neighbors (ENN) to remove ambiguous examples.
    # sampling_strat = dict(Counter(y.ravel()))
    # under = NeighbourhoodCleaningRule(kind_sel='mode', n_jobs=threads)
    # under = ClusterCentroids(random_state=rand, sampling_strategy=sampling_strat, n_jobs=threads)  # too slow
    # under = TomekLinks(n_jobs=threads)
    # under = NearMiss(version=1, sampling_strategy='all', n_jobs=threads) #too radical
    # under = EditedNearestNeighbours(kind_sel='mode', n_jobs=threads)
    # X, y = under.fit_resample(X, y)

    sampling_strat = dict(Counter(y.ravel()))
    print('Resample results for training only:\n {}'.format(sampling_strat))
    return X, y


# #############################################################################
# Preprocess and reuse same variables (dirty but concise.. I do this everywhere)
def preproc(X_train, X_test, y_train, y_test, specified_scaler, feature_selection, feature_limit=21):
    # Scale feature values
    if specified_scaler == "none":
        scaler = Normalizer()
    if specified_scaler == "standard":
        scaler = StandardScaler()  # -- works well when outliers are negligible
    # scaler = MinMaxScaler(feature_range = (0, 10)) # -- works well when outliers are negligible
    # Robust ignores (no pruned) small and large outliers, given a percentile and scales rest of data
    if specified_scaler == "robust":
        scaler = RobustScaler(quantile_range=(25, 75))
    # quantile transformers changes the distribution and even makes outliers part of inliers-- good for uniform data
    if specified_scaler == "quantile":
        scaler = QuantileTransformer(output_distribution='uniform')  # fast and great
    # PowerTransformer finds the optimal scaling factor to stabilize variance through maximum likelihood estimation
    if specified_scaler == "power":
        scaler = PowerTransformer(method='yeo-johnson')  # great
    # Fit on training set only
    scaler.fit(X_train[feature_selection[:feature_limit]])

    # Apply transform to both the training set and the test set and standard naming conventions
    X_train = scaler.transform(X_train[feature_selection[:feature_limit]])
    X_test = scaler.transform(X_test[feature_selection[:feature_limit]])
    return X_train, X_test, y_train, y_test


# #############################################################################
# Feature Extraction: random state 42 is used for all stochastic algorithms
def f_extract(X_train, X_test, y_train, y_test, method='26PCA', feature_limit=26):
    def str_split_num(s):
        tail = s.lstrip('0123456789')  # use rstrip if num is last part of str
        head = s[0:-len(tail)]  # negative to count from last char
        return int(head), tail

    if method[0].isdigit():
        n_comps, method = str_split_num(method)
        print("Feature extraction using", method)

    if method == 'PCA':
        reducer = PCA(n_components=n_comps, whiten=True, random_state=rand).fit(X_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    if method == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=n_comps).fit(X_train, y_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    if method == 'ICA':
        reducer = FastICA(n_components=n_comps, whiten=True, random_state=rand).fit(X_train, y_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    if method == 'LLE':  # too slow
        reducer = LocallyLinearEmbedding(n_components=n_comps, random_state=rand, n_jobs=threads).fit(X_train, y_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    if method == 'TSNE':
        reducer = TSNE(n_components=n_comps, learning_rate=1000, metric='euclidean', n_iter=11, random_state=rand, n_jobs=threads).fit(X_train, y_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    if method == 'UMAP':  # too slow ...angular_rp_forest=True,
        y_train, y_test = encode_labels(y_train, y_test)
        reducer = UMAP(n_components=n_comps, n_neighbors=15, metric='correlation', random_state=rand, min_dist=0.0,
                       angular_rp_forest=True, n_epochs=15).fit(X_train, y_train)
        X_train = reducer.transform(X_train)
        X_test = reducer.transform(X_test)

    return X_train, X_test


# #############################################################################
# create rnn model
def create_rnn(fnum=26, cnum=15, learn_rate=0.0001, dropout_rate=0.2, neurons=240):
    print(fnum, cnum)
    rnn = Sequential()
    rnn.add(SimpleRNN(neurons, input_shape=(1, fnum), return_sequences=True))
    rnn.add(Dropout(dropout_rate))

    rnn.add(SimpleRNN(neurons, return_sequences=True))
    rnn.add(Dropout(dropout_rate))

    rnn.add(SimpleRNN(neurons, return_sequences=False))
    rnn.add(Dropout(dropout_rate))
    # binary class prediction layer
    # rnn.add(Dense(1))
    # rnn.add(Activation('sigmoid'))
    # multiclass prediction layer
    rnn.add(Dense(cnum))  # 15 classes (outputs)
    rnn.add(Activation('softmax'))
    # rnn.summary()
    # optimizer
    adam = Adam(lr=learn_rate)

    # binary
    # rnn.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])
    # multiclass
    rnn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return rnn

# #############################################################################
# create gru-lstm model (perhaps requires less neurons than simpleRNN)
def create_gru_l(fnum=26, cnum=15, learn_rate=0.0001, dropout_rate=0.2, neurons=240):
    print(fnum, cnum)

    grul = Sequential()
    grul.add(GRU(neurons, input_shape=(1, fnum), return_sequences=True))
    grul.add(Dropout(dropout_rate))

    grul.add(Dense(neurons, activation='relu'))
    grul.add(Dropout(dropout_rate))

    grul.add(LSTM(neurons, return_sequences=True))
    grul.add(Dropout(dropout_rate))

    grul.add(Dense(neurons, activation='relu'))
    grul.add(Dropout(dropout_rate))

    grul.add(GRU(neurons, return_sequences=False))
    grul.add(Dropout(dropout_rate))

    # multiclass prediction layer
    grul.add(Dense(cnum))  # 15 classes (outputs)
    grul.add(Activation('softmax'))
    # rnn.summary()
    # optimizer
    adam = Adam(lr=learn_rate)

    # binary
    # rnn.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])
    # multiclass
    grul.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return grul

# #############################################################################
# create cnn (perhaps requires less neurons than simpleRNN)
def create_cnn(fnum=26, cnum=15, learn_rate=0.0001, dropout_rate=0.2, neurons=240):
    print(fnum, cnum)

    cnn = Sequential()
    cnn.add(Conv1D(neurons, kernel_size=1, activation='relu', input_shape=(1, fnum)))
    cnn.add(Dropout(dropout_rate))

    # cnn.add(Conv1D(neurons, kernel_size=1, activation='relu'))
    # cnn.add(Dropout(dropout_rate))
    #
    # cnn.add(Conv1D(neurons, kernel_size=1, activation='relu'))
    # cnn.add(Dropout(dropout_rate))
    #
    # cnn.add(Flatten())

    # multiclass prediction layer
    cnn.add(Dense(cnum))  # 15 classes (outputs)
    cnn.add(Activation('softmax'))
    # rnn.summary()
    # optimizer
    adam = Adam(lr=learn_rate)

    # binary
    # rnn.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])
    # multiclass
    cnn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn

# #############################################################################
# create lrg_cnn model (lstm, rnn, gru and flattened as input for cnn layer)
def create_lrg_cnn(fnum=26, cnum=15, learn_rate=0.0001, dropout_rate=0.2, neurons=240):
    print(fnum, cnum)

    input_1 = Input(name='left_input', shape=(1, fnum))
    input_2 = Input(name='middle_input', shape=(1, fnum))
    input_3 = Input(name='right_input', shape=(1, fnum))

    left = LSTM(neurons, return_sequences=True)(input_1)
    left = LSTM(neurons, return_sequences=True)(left)
    left = LSTM(neurons, return_sequences=True)(left)
    left = Dropout(dropout_rate)(left)

    middle = SimpleRNN(neurons, return_sequences=True)(input_2)
    middle = SimpleRNN(neurons, return_sequences=True)(middle)
    middle = SimpleRNN(neurons, return_sequences=True)(middle)
    middle = Dropout(dropout_rate)(middle)

    right = GRU(neurons, return_sequences=True)(input_3)
    right = GRU(neurons, return_sequences=True)(right)
    right = GRU(neurons, return_sequences=True)(right)
    right = Dropout(dropout_rate)(right)

    merged = Concatenate(axis=-1)([left, middle, right])

    final = Conv1D(neurons, kernel_size=1)(merged)
    final = Flatten()(final)

    # multiclass prediction layer
    # lrgc.add(Dense(cnum))  # 15 classes (outputs)
    # lrgc.add(Activation('softmax'))
    predictions = Dense(cnum, name='prediction_layer', activation='softmax')(final)
    lrgc = Model(inputs=[input_1, input_2, input_3], outputs=predictions)
    # rnn.summary()
    # optimizer
    adam = Adam(lr=learn_rate)

    # binary
    # rnn.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])
    # multiclass
    lrgc.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return lrgc


# #############################################################################
# Grid of parameters which could be optimal per classifier
def classifier_tuner(X_train, X_test, y_train, y_test, specified_classifier, specified_fnum, specified_scaler,
                     method, train_size=0.5):
    # ***********************************KNN grid***************************************************************************
    if specified_classifier == "knn":
        tune_params = {'n_neighbors': [3, 5], 'metric': ['manhattan'], 'weights': ['distance', 'uniform']}
        clf = KNeighborsClassifier(n_jobs=threads)

    # ***********************************Random Forest grid***************************************************************************
    elif specified_classifier == "rf":
        n_estimators = [int(x) for x in np.linspace(start=300, stop=600, num=2)]  # no. trees in forest
        max_features = ['sqrt', 'log2']  # no. features per split
        # max_depth = [int(x) for x in np.linspace(25, 50, num=2)]  # levels per split
        max_depth = [30]  # levels per split
        min_samples_split = [3]  # Min samples to split a node
        min_samples_leaf = [3]  # Min samples at each leaf node
        bootstrap = [True, False]  # sampling per tree vs all samples

        # 'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 110, 'bootstrap': False
        # {'n_estimators': 1000, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_depth': None}
        tune_params = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       # 'bootstrap': bootstrap,
                       'max_depth': max_depth,
                       # 'min_samples_split': min_samples_split,
                       # 'min_samples_leaf': min_samples_leaf
                       "criterion": ["gini", "entropy"]
                       }

        clf = RandomForestClassifier(random_state=rand, warm_start=True, n_jobs=threads)

    # ***********************************MLP grid***************************************************************************
    elif specified_classifier == "mlp":
        # {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (50, 50, 50), 'alpha': 0.0001, 'activation': 'tanh'}
        tune_params = [
            {'hidden_layer_sizes': [(10, 10, 10), (100, 100, 100), (200, 200, 200)], 'activation': ['relu'],
             'solver': ['adam'], 'alpha': [0.0001], 'learning_rate': ['constant']}
        ]
        clf = MLPClassifier(random_state=rand, warm_start=True)

    # ***********************************Linear SVM Grid***************************************************************************
    elif specified_classifier == "svm":
        tune_params = [{'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'C': [100, 200], 'class_weight': [None]},
                       {'penalty': ['l2'], 'loss': ['hinge'], 'dual': [True], 'C': [100, 200], 'class_weight': [None]}
                       ]
        clf = LinearSVC(random_state=rand)

    # ***********************************RBF SVM Grid***************************************************************************
    elif specified_classifier == "ksvm":  # mid C low gamma except for quantile, probably need PCA
        gamma_lst = [0.5, 1, 3]
        C_lst = [25, 100]
        if method != 'none':  # extracted methods make data more linear, so smaller gamma
            gamma_lst = [element / 10 for element in gamma_lst]
        if train_size >=0.3:  # ksvm scaling issue, so more generalization
            gamma_lst = [element * (1-train_size) / 2 for element in gamma_lst]
            C_lst = [element * (1-train_size) for element in C_lst]

        if specified_scaler == "quantile" or "power":
            tune_params = [{'kernel': ['rbf'], 'gamma': gamma_lst, 'C': C_lst, 'random_state': [rand], 'class_weight': [None]}]
        if specified_scaler == "robust":
            gamma_lst = [element /2 for element in gamma_lst]
            tune_params = [{'kernel': ['rbf'], 'gamma': gamma_lst, 'C': C_lst, 'random_state': [rand], 'class_weight': [None]}]
        if specified_scaler == "standard" or "none":
            tune_params = [{'kernel': ['rbf'], 'gamma': gamma_lst, 'C': C_lst, 'random_state': [rand], 'class_weight': [
                None]}]

        clf = SVC(random_state=rand, probability=True, cache_size=1000)
        # clf = OneVsRestClassifier(SVC(kernel='linear', probability=True)
        # clf = OneVsRestClassifier(BaggingClassifier(base_estimator=SVC(kernel='linear', probability=True),
        #                         max_samples=1.0 / n_estimators, n_estimators=n_estimators))

    # ***********************************Gaussian Naive Bayes Grid***************************************************************************
    elif specified_classifier == "gnb":
        tune_params = [{'priors': [None]}
                       ]
        clf = GaussianNB()


    # ***********************************Decision Tree Grid***************************************************************************
    elif specified_classifier == "dt":
        max_features = ['sqrt', 'log2']  # no. features per split
        # max_depth = [int(x) for x in np.linspace(25, 50, num=2)]  # levels per split
        max_depth = [30]  # levels per split
        min_samples_split = [3]  # Min samples to split a node
        min_samples_leaf = [3]  # Min samples at each leaf node

        tune_params = [{"max_depth": max_depth,
                        "max_features": max_features,
                        "min_samples_leaf": min_samples_leaf,
                        "min_samples_split": min_samples_split,
                        "criterion": ["gini", "entropy"]}
                       ]

        clf = DecisionTreeClassifier(random_state=rand)

    # ***********************************Logistic Regresion Grid*************************************************************************************
    elif specified_classifier == "lr":

        tune_params = [{'C': [10, 100, 400, 800]}
                       ]

        clf = LogisticRegression(random_state=rand, warm_start=True, n_jobs=threads)

    # ***********************************RNN Grid*************************************************************************************
    elif specified_classifier == "rnn":
        # create model
        clf = KerasClassifier(build_fn=create_rnn, fnum=X_train.shape[1], cnum=len(np.unique(y_test)), verbose=0)  #
        # wrapper for scikit?
        # define the grid search parameters
        neurons = [128, 256]
        batch_size = [64, 128]
        epochs = [1000]
        learn_rate = [0.0005]
        # weight_constraint = [1, 3, 5]
        dropout_rate = [0.1]
        momentum = [0.0, 0.3, 0.6, 0.9]
        tune_params = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, dropout_rate=dropout_rate)

        # ***********************************GRU-LSTM Grid****************************************************************************
    elif specified_classifier == "grul":
        # create model
        clf = KerasClassifier(build_fn=create_gru_l, fnum=X_train.shape[1], cnum=len(np.unique(y_test)),
                              verbose=0)  # wrapper for scikit
        # define the grid search parameters
        neurons = [128, 256]
        batch_size = [64, 128]
        epochs = [1000]
        learn_rate = [0.0001]
        # weight_constraint = [1, 3, 5]
        dropout_rate = [0.2]
        momentum = [0.0, 0.3, 0.6, 0.9]
        tune_params = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
                           dropout_rate=dropout_rate)

        # ***********************************CNN Grid****************************************************************************
    elif specified_classifier == "cnn":
        # create model
        clf = KerasClassifier(build_fn=create_lrg_cnn, fnum=X_train.shape[1], cnum=len(np.unique(y_test)),
                              verbose=0)  # wrapper for scikit
        # define the grid search parameters
        neurons = [128, 256]
        batch_size = [64, 128]
        epochs = [2]
        learn_rate = [0.001]
        # weight_constraint = [1, 3, 5]
        dropout_rate = [0.2]
        momentum = [0.0, 0.3, 0.6, 0.9]
        tune_params = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
                           dropout_rate=dropout_rate)

    # ***********************************LSTM-RNN-GRU-CNN # ***********************************GRU-LSTM Grid****************************************************************************
    #     elif specified_classifier == "lrgc":
    #         # create model
    #         clf = KerasClassifier(build_fn=create_lrg_cnn, fnum=X_train.shape[1], cnum=len(np.unique(y_test)),
    #                               verbose=0)  # wrapper for scikit
    #         # define the grid search parameters
    #         neurons = [128, 256]
    #         batch_size = [64, 128]
    #         epochs = [2]
    #         learn_rate = [0.001]
    #         # weight_constraint = [1, 3, 5]
    #         dropout_rate = [0.2]
    #         momentum = [0.0, 0.3, 0.6, 0.9]
    #         tune_params = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
    #                            dropout_rate=dropout_rate) Grid****************************************************************************
    elif specified_classifier == "lrgc":
        # create model
        clf = KerasClassifier(build_fn=create_lrg_cnn, fnum=X_train.shape[1], cnum=len(np.unique(y_test)),
                              verbose=0)  # wrapper for scikit
        # define the grid search parameters
        neurons = [128, 256]
        batch_size = [64, 128]
        epochs = [2]
        learn_rate = [0.001]
        # weight_constraint = [1, 3, 5]
        dropout_rate = [0.2]
        momentum = [0.0, 0.3, 0.6, 0.9]
        tune_params = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
                           dropout_rate=dropout_rate)

    # ***********************************Bagging version of any classifier***************************************************************************
    n_estimators = 10
    bag_clf = OneVsRestClassifier(BaggingClassifier(random_state=rand, base_estimator=clf, max_samples=1.0 / n_estimators,
                                                    n_estimators=n_estimators, n_jobs=threads))
    return tune_params, clf, bag_clf


# #############################################################################
# Classifier selection
def classifier_select(X_train, X_test, y_train, y_test, specified_classifier, specified_scaler, n_estimators=10, n_jobs=2):
    # Grid of parameters which could be optimal per classifier
    if specified_classifier == "knn":
        clf = KNeighborsClassifier(weights='distance', metric='manhattan', n_jobs=n_jobs)

    elif specified_classifier == "rf":
        clf = RandomForestClassifier(random_state=rand, warm_start=True, n_jobs=n_jobs)

    elif specified_classifier == "svm":
        clf = LinearSVC(random_state=rand)  # multi class

    elif specified_classifier == "ksvm":  # mid C low gamma except for quantile, probably need PCA
        clf = SVC(random_state=rand, kernel='rbf', probability=True, gamma=0.1, C=100, cache_size=1000)

    elif specified_classifier == "mlp":
        clf = MLPClassifier(random_state=rand, warm_start=True, hidden_layer_sizes=(100, 100, 100))

    elif specified_classifier == "gnb":
        clf = GaussianNB()

    elif specified_classifier == "dt":
        clf = DecisionTreeClassifier(random_state=rand)

    elif specified_classifier == "lr":
        clf = LogisticRegression(random_state=rand, warm_start=True, n_jobs=threads)

    elif specified_classifier == "rnn":
        clf = KerasClassifier(build_fn=create_rnn, fnum=X_train.shape[2], cnum=len(np.unique(y_test)), neurons=128, batch_size=64, epochs=10, dropout_rate=0.2,
                              verbose=0)  # wrapper for scikit?

    elif specified_classifier == "grul":
        clf = KerasClassifier(build_fn=create_gru_l, fnum=X_train.shape[2], cnum=len(np.unique(y_test)), neurons=128,
                              batch_size=64, epochs=10, dropout_rate=0.2,
                              verbose=0)  # wrapper for scikit?

    elif specified_classifier == "cnn":
        clf = KerasClassifier(build_fn=create_cnn, fnum=X_train.shape[2], cnum=len(np.unique(y_test)), neurons=64,
                              batch_size=64, epochs=10, dropout_rate=0.1,
                              verbose=0)  # wrapper for scikit?

    elif specified_classifier == "lrgc":
        clf = KerasClassifier(build_fn=create_lrg_cnn, fnum=X_train.shape[2], cnum=len(np.unique(y_test)), neurons=128,
                              batch_size=64, epochs=10, dropout_rate=0.2,
                              verbose=0)  # wrapper for scikit?

    bag_clf = OneVsRestClassifier(
        BaggingClassifier(random_state=rand, base_estimator=clf, max_samples=1.0 / n_estimators, n_estimators=n_estimators,
                          n_jobs=n_jobs))
    return clf, bag_clf


# class oversampled_Kfold():
#     def __init__(self, n_splits, n_repeats=1):
#         self.n_splits = n_splits
#         self.n_repeats = n_repeats
#
#     def get_n_splits(self, X, y, groups=None):
#         return self.n_splits*self.n_repeats
#
#     def split(self, X, y, groups=None):
#         splits = np.split(np.random.choice(len(X), len(X),replace=False), 5)
#         train, test = [], []
#         for repeat in range(self.n_repeats):
#             for idx in range(len(splits)):
#                 trainingIdx = np.delete(splits, idx)
#                 Xidx_r, y_r = ros.fit_resample(trainingIdx.reshape((-1,1)),
# y[trainingIdx])
#                 train.append(Xidx_r.flatten())
#                 test.append(splits[idx])
#         return list(zip(train, test))
# ...
# ...
# rkf_search = oversampled_Kfold(n_splits=5, n_repeats=2)
# ...
# output = cross_validate(clf,x,y, scoring=metrics,cv=rkf)
# # Where ros was the Random oversampler from imblearn.


# #############################################################################
# For model saving and loading:
def load_train(load_model_path, clf, X_train, y_train, tune_params=None, use_smote=0, n_jobs=1):
    grid = 'none'
    vld_sz = 0.5
    if not os.path.isfile(load_model_path):
        if tune_params == None:
            print("Default training...")
            clf.fit(X_train, y_train)
        else:
            print("NO EXISTO... Performing Grid Search to train model using CV=5 or 30% vld...")

            # predefined 90/10 train/vld split and resample
            X_train_sm, X_val, y_train_sm, y_val = train_test_split(X_train, y_train, test_size=vld_sz, random_state=rand)
            # X_train_sm = X_train_sm.reshape(X_train_sm.shape[0], X_train_sm.shape[2]) #uncomment for Keras clf
            X_train_sm, y_train_sm = resample(X_train_sm, y_train_sm, use_smote=use_smote)
            # X_train_sm = X_train_sm.reshape(X_train_sm.shape[0], 1, X_train_sm.shape[1]) #uncomment for Keras clf

            # train data has label -1 and 20% of that is reassigned to validation
            split_index = np.repeat(-1, y_train_sm.shape)
            np.random.seed(rand)
            split_index[np.random.choice(split_index.shape[0], int(round(vld_sz * split_index.shape[0])), replace=False)] = 0
            cv = list(PredefinedSplit(split_index).split())

            # cross validation cv=5 = 80/20
            # cv = 5

            if not type(clf).__name__ == 'KerasClassifier':
                f1_metric = metrics.make_scorer(metrics.f1_score, average='macro')
                # grid = RandomizedSearchCV(clf, scoring=f1_metric, param_distributions=tune_params, verbose=1, n_iter=20, cv=3, n_jobs=n_jobs)
                grid = GridSearchCV(clf, scoring=f1_metric, param_grid=tune_params, cv=cv, n_jobs=n_jobs)
                # grid = GridSearch(model=clf, param_grid=tune_params, scoring=f1_metric)
            else:
                grid = GridSearchCV(clf, param_grid=tune_params, cv=cv, n_jobs=n_jobs)
                # grid = GridSearch(model=clf, param_grid=tune_params)

            if type(clf).__name__ == 'KerasClassifier':
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # add timesteps
                y_train = np_utils.to_categorical(y_train)  # multiclass
                # X_train = [X_train, X_train, X_train] # todo: for cnn
            grid.fit(X_train, y_train)

            print(grid.best_params_)

            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            # Save model
            joblib.dump(grid, load_model_path, compress=6)
            print("model saved... I think\n")
            modelLoad = grid
    else:
        print("already exists...loading model")
        modelLoad = joblib.load(load_model_path)
        # modelLoad.best_estimator_.named_steps['svm']
        # print(modelLoad.classes_)

    return grid, modelLoad


# #############################################################################
# Classify using all processors (change n_jobs to 1 if you have problems with memory/lockup etc.):
def parallel_classify(modelLoad, X_test, y_test, n_jobs=1, probability=False, pca=False):
    from joblib import Parallel, delayed
    from sklearn.utils import gen_batches

    n_samples, n_features = X_test.shape
    batch_size = n_samples // n_jobs

    # fastest (might be unsafe)
    def _predict(method, X, sl):
        return method(X[sl])

    if probability == False:
        y_pred_list = Parallel(n_jobs)(delayed(_predict)(modelLoad.predict, X_test, sl)
                                       for sl in gen_batches(n_samples, batch_size))
    else:
        y_pred_list = Parallel(n_jobs)(delayed(_predict)(modelLoad.predict_proba, X_test, sl)
                                       for sl in gen_batches(n_samples, batch_size))

    y_pred = np.asarray(list(chain.from_iterable(y_pred_list)))  # 9D list of arrays to a 1D numpy array
    return y_pred
