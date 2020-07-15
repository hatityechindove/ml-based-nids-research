""" Usage (example):
python Train_Test.py --dataset cicids2017_int_class.csv --classifier rnn --features 3 --fnum 69 --scale
 quantile --extract none --multiclass 1 --trainsize 0.1 --testsize 0.9  >>  keras_results.txt 2>&1 &
"""

# #############################################################################
# Parameters usage:
import os
import sys
import time
import warnings
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import np_utils

from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from Common import load_all_data, pd_feature_sel, getCommandArgs, preproc, classifier_tuner, encode_labels, load_train, \
    parallel_classify, fast_load_data, sample_cnt, f_extract
from Visualization.Visualize_Data import plot_learning_curve

random_state = 42
if not sys.warnoptions:  # this is dangerous and a quickfix for annoying LinearSVC not converging
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

from multiprocessing import cpu_count

threads = cpu_count()//2 #hyperthreading/SMT is not worth it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# #############################################################################
# Command line arguments (specify default):
spec_dataset, spec_classifier, spec_fsel, spec_fnum, spec_scaler, spec_fext, multiclass, train_size, test_size, use_sm = getCommandArgs(
    'cicids2017_int_class.csv', 'gnb', 1, 26, 'quantile', 'none', 1, 0.1, 0.9, 1)

# spec_dataset, spec_classifier, spec_fsel, spec_fnum, spec_scaler, spec_fext, multiclass, train_size, test_size, use_sm = getCommandArgs(
#     'cicids2018_int_class.csv', 'gnb', 1, 26, 'quantile', 'none', 1, 0.1, 0.9, 1)

print("Classifying with:", spec_classifier)
print("Feature count:", spec_fnum)
print("Scaling with:", spec_scaler)
class_type = "Multi" if multiclass == 1 else "Binary"
print("Multiclass?", class_type)
print("Loading and Splitting train_size =", train_size * 100, '% of dataset')

# #############################################################################
# Load data or model from a path:
start_test = time.time()
data_path = '/mnt/ProgsNBack/Work/Datasets/hati_data/csvs/'  # contains hati's combined CSVs
model_path = '/mnt/ProgsNBack/Work/Datasets/hati_data/models/'
# data = load_all_data(data_path, 'concat.csv')  # hati's cicids2017
# data = load_all_data(data_path, 'cicids2017.csv')  # cicids2017 orig
# data = load_all_data(data_path, dataset)  # cicids2017 orig
data = fast_load_data(data_path + spec_dataset, dtype='int', ex_class=[])  # cicids2017 int
# data['Label'] = data['Label'].replace("DDOS", "DDoS", regex=True) #specified column(s)

# #############################################################################
# Define feature selectors and input the train/test split ratio
X_train, X_test, y_train, y_test, feature_selection = pd_feature_sel(data, train_size, test_size, multiclass,stratify=True, orig_labels=True)
end_test = time.time()
print("Data load + sample time=",end_test-start_test)
# #############################################################################
# Preprocess all data according to train/test split
start_test = time.time()
X_train, X_test, y_train, y_test = preproc(X_train, X_test, y_train, y_test, spec_scaler, feature_selection[
    spec_fsel], feature_limit=spec_fnum)

# #############################################################################
# Feature Extraction (LDA max comp = classes -1)
X_train, X_test = f_extract(X_train, X_test, y_train, y_test, method=spec_fext)
end_test = time.time()
print("Feature processing time=",end_test-start_test)

# #############################################################################
# Encode train and test labels (necessary for certain classfiers like RF and neuralnets)
y_train_enc, y_test_enc = encode_labels(y_train, y_test)
y_train_enc_int = y_train_enc #to keep count number of classes (non-categorical)
y_test_enc_int = y_test_enc
# #############################################################################
# Deep learning algorithms: re-encode data
if spec_classifier in {'rnn', 'grul'}:
    # reshape input to be [samples, timesteps(aka sequence legnth), features]
    # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) #add timesteps
    # y_train_enc = np_utils.to_categorical(y_train_enc) # multiclass
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test_enc = np_utils.to_categorical(y_test_enc)

# #############################################################################
# Grid of parameters for classifier tuning
tune_params, clf, bag_clf = classifier_tuner(X_train, X_test, y_train_enc_int, y_test_enc_int, spec_classifier,
                                             spec_fnum, spec_scaler, method=spec_fext, train_size=train_size)

# #############################################################################
# For model saving and loading (with timing):
if spec_dataset == 'cicids2017_int_class.csv.zip':
    set = spec_dataset[:-8]
else:
    set = spec_dataset[:-4]

if spec_fext =='none':
    spec_fext =""
start_train = time.time()
if multiclass == 1:
    load_model_path = model_path + set + spec_classifier + str(spec_fsel) + "_" + str(spec_fnum) + spec_scaler + spec_fext + str(train_size) + 'mc.pkl'
else:
    load_model_path = model_path + set + spec_classifier + str(spec_fsel) + "_" + str(spec_fnum) + spec_scaler + spec_fext + str(train_size) + 'bc.pkl'

grid, modelLoad = load_train(load_model_path, clf, X_train, y_train_enc, tune_params, use_smote=use_sm, n_jobs=threads)
end_train = time.time()

print("Multiclass" if multiclass == 1 else "Binaryclass", "train time for", spec_classifier,
      spec_scaler, "of trainsize=",
      train_size, "is", round(end_train - start_train, 2), "s")

class_size = len(np.unique(y_test))
print(class_size)
# #############################################################################
# Classify using all processors (change n_jobs to 1 if you have problems with memory/lockup etc.):
start_test = time.time()
if not type(clf).__name__=='KerasClassifier': #traditional model
    if not type(clf).__name__ == 'RandomForestClassifier':
        # print(type(clf).__name__)
        y_pred = parallel_classify(modelLoad, X_test, y_test_enc, n_jobs=threads, probability=False)
    else:
        y_pred = modelLoad.predict(X_test)

    # print(np.unique(y_test))
    c_report = metrics.classification_report(y_test_enc, y_pred, target_names=np.unique(y_test))
    # c_report = metrics.classification_report(y_test_enc, y_pred, target_names=np.unique(y_test), labels=list(range(0, class_size)))
    # c_report = metrics.classification_report(y_test_enc, y_pred, target_names=np.unique(y_test), labels=[0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14])

else: #DL model
    y_pred = modelLoad.predict(X_test)
    c_report = metrics.classification_report(y_test_enc_int, y_pred, target_names=np.unique(y_test))
    # c_report = metrics.classification_report(y_test_enc_int, y_pred, target_names=np.unique(y_test), labels=list(range(0, class_size)))

    # c_report = metrics.classification_report(y_test_enc_int, y_pred, target_names=np.unique(y_test_enc_int), labels=[0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14])



end_test = time.time()
print("\nClassification report - \n", c_report)
print("Multiclass" if multiclass == 1 else "Binaryclass", "test time for", type(clf).__name__, spec_scaler,
      "of trainsize=",train_size, "is", round(end_test - start_test, 2), "s")


print("-------------------------------------***DONE***-------------------------------------------")
print("-------------------------------------$$$NEXT$$$-------------------------------------------")