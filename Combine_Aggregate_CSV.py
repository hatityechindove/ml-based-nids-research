import argparse
import os
import glob
import functools
import pandas as pd
import numpy as np
import re
# python Combine_Aggregate_CSV.py --dir hati_data/cicids2018/ #all csv files in that dir
# python Combine_Aggregate_CSV.py --dir hati_data/cicids2018/ --files Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv Friday-WorkingHours-Morning.pcap_ISCX.csv
# #############################################################################
# # Command line arguments
parser = argparse.ArgumentParser(description='Various parameters can be specified')
parser.add_argument('--files', default="none", nargs='+')
parser.add_argument('--dir', default="none")
parser.add_argument('--out', default="combined.csv")
parser.add_argument('--reclass', type=int, default=0)
args = parser.parse_args()
files_path = args.files
dir_path = args.dir
out = args.out
reclass = args.reclass

if not dir_path.endswith("/"):
    dir_path += "/"

# #############################################################################
# # Load data or model from a path:
def load_all(dir_path, files_path):
    if files_path == "none":
        print("\nno files specified..doing all csvs in", dir_path, "\n\n")
        files = glob.glob(dir_path + '*.csv')
        print(files)
    else:
        files = [dir_path + s for s in files_path]  # list comprehension is boss
        print(files)

    combined = pd.concat(map(functools.partial(pd.read_csv, engine='c', dtype='object'), files))
    return combined

def load_lite_data(dir_path, files_path, non_numeric=['Label']):
    if files_path == "none":
        print("\nno files specified..doing all csvs in", dir_path, "\n\n")
        files = glob.glob(dir_path + '*.csv')
        print(files)
    else:
        files = [dir_path + s for s in files_path]  # list comprehension is boss
        print(files)

    cols_to_skip = non_numeric

    if len(files) > 1: 
        df_str = [pd.read_csv(f, engine='c', dtype='object', usecols=lambda x: x in cols_to_skip) for f in files]
        cols_to_skip.extend(['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Protocol', 'Timestamp', 'imbalance_label', 'is_malicious'])  # remove these columns
        df_num = [pd.read_csv(f, engine='c', dtype='float', usecols=lambda x: x not in cols_to_skip) for f in files]

        df_num = pd.concat(df_num, ignore_index=True) #ignore index so that it continues counting indices as if it was originally one csv
        df_str = pd.concat(df_str, ignore_index=True)
        concat = pd.concat([df_num, df_str], axis=1) #add the label column(s) to the rest
    else:
        df_str = pd.read_csv(files[0], engine='c', dtype='object', usecols=lambda x: x in cols_to_skip)
        cols_to_skip.extend(['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Protocol', 'Timestamp', 'imbalance_label', 'is_malicious'])  # remove these columns
        df_num = pd.read_csv(files[0], engine='c', dtype='float', usecols=lambda x: x not in cols_to_skip)
        concat = pd.concat([df_num, df_str], axis=1)  # add the label column(s) to the rest
    return concat

# #############################################################################
# # Merge network classes for cicids2017:
def aggregate_net_class2017(data):
    # data.loc[data['Label'] == 'BENIGN', 'imbalance_label'] = 'Normal'
    # data.loc[data['Label'] == 'Bot', 'imbalance_label'] = 'U2R'
    # data.loc[data['Label'] == 'DDoS', 'imbalance_label'] = 'DoS'
    # data.loc[data['Label'] == 'DoS GoldenEye', 'imbalance_label'] = 'DoS'
    # data.loc[data['Label'] == 'DoS Hulk', 'imbalance_label'] = 'DoS'
    # data.loc[data['Label'] == 'DoS Slowhttptest', 'imbalance_label'] = 'DoS'
    # data.loc[data['Label'] == 'DoS slowloris', 'imbalance_label'] = 'DoS'
    # data.loc[data['Label'] == 'FTP-Patator', 'imbalance_label'] = 'R2L'
    # data.loc[data['Label'] == 'Heartbleed', 'imbalance_label'] = 'R2L'
    # data.loc[data['Label'] == 'Infiltration', 'imbalance_label'] = 'U2R'
    # data.loc[data['Label'] == 'PortScan', 'imbalance_label'] = 'Probing'
    # data.loc[data['Label'] == 'SSH-Patator', 'imbalance_label'] = 'R2L'
    data.loc[data['Label'].str.contains('Benign', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Normal'
    data.loc[data['Label'].str.contains('DDoS|Bot', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'DDoS/Bot'
    data.loc[data['Label'].str.contains('Infiltration', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Infiltration'
    data.loc[data['Label'].str.contains('FTP|SSH', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Brute'
    data.loc[data['Label'].str.contains('Web', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Web'
    data.loc[data['Label'].str.contains('\bDoS', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'DoS'
    data.loc[data['Label'].str.contains('Heartbleed', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Heartbleed'
    data.loc[data['Label'].str.contains('Port', flags=re.IGNORECASE, regex=True), 'imbalance_label'] = 'Probing'
    return data

def get_dups(data):
    print('Cleaning data...')
    duplicateRows = data[data.duplicated()]
    print("Duplicate Rows except first occurrence are :")
    dups = duplicateRows.shape[0]
    if dups > 0:
        print(dups)
        return data.drop_duplicates().dropna()
    else:
        print("no dups")
        return data
# #############################################################################
# # Group classes for binary classification
def add_binary_net_class(data):

    clean_up = get_dups(data)
    clean_up.loc[clean_up['imbalance_label'] == 'Normal', 'is_malicious'] = 'No'
    clean_up.loc[clean_up['imbalance_label'] != 'Normal', 'is_malicious'] = 'Yes'
    return clean_up

def resample_net(clean_up):
    benign_sample = clean_up[clean_up['imbalance_label'] == 'Normal'].sample(n=len(clean_up[clean_up['imbalance_label'] !='Normal']), random_state=42)
    clean_up = clean_up[clean_up['imbalance_label'] != 'Normal']
    preprocess_data = pd.concat([clean_up, benign_sample])
    return preprocess_data

def rename_cols(data):

    # data.columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
    #                 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    #                 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    #                 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
    #                 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
    #                 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    #                 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
    #                 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
    #                 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    #                 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
    #                 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    #                 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    #                 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
    #                 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

    data.columns = ['Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
                    'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
                    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
                    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                    'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
                    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                    'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
                    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                    'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
                    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
                    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
                    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
    return data.columns
# #############################################################################
# ***************************** MAIN ******************************************
print('Loading data')
# data = load_lite_data(dir_path, files_path, non_numeric=['Label', 'imbalance_label', 'is_malicious'])
# data = load_lite_data(dir_path, files_path, non_numeric=['Label'])
data = load_lite_data(dir_path, files_path, non_numeric=['Label']) #cicids2018
# data = load_all(dir_path, files_path)

# print('Renaming Features..')
data.columns = rename_cols(data) #name columns concisely and consistently

# print('Formatting data..dropping useless columns (features)')
# data = data.drop(data.columns[[0, 1, 2, 3, 6, 61]], axis=1) #cicids2017 Labelled flows
# data = data.drop(data.columns[[55]], axis=1) #cicids2017 ML
# data = data.drop(data.columns[[2]], axis=1) #cicids2018

# Remove certain characters:
# data.columns = data.columns.str.strip()  # remove annoying leading/trailing whitespaces in headers
# data = data.apply(lambda x: x.str.replace("\,|'|\-|\/|\:|\,| ", "", regex=True) if x.dtype == "object" else x)
# data = data.apply(lambda x: x.str.replace("\,| ", ".", regex=True) if x.dtype == "object" else x)
# data = data.apply(lambda x: x.str.replace("�", "-", regex=True) if x.dtype == "object" else x)

def remove_non_ascii(text,replace_with=''):
    return replace_with.join(i for i in text if ord(i)<128)
data['Label'] = data['Label'].apply(remove_non_ascii)

# print('removing column names dup in cicids2018') # remove this
# data.drop(data.loc[data['Dst Port'].str.contains('Dst Port|DstPort|Dst', flags=re.IGNORECASE, regex=True)].index, inplace=True) #cicids2018

print('renaming: classes in cicids2018') #cicids2018
data['Label'] = data['Label'].replace("Brute Force -Web", "Web Brute Force", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("Brute Force -XSS", "Web XSS", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DDOS attack-HOIC", "DDoS", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DDOS attack-LOIC-UDP", "DDoS", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DoS attacks-GoldenEye", "DoS GoldenEye", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DoS attacks-Hulk", "DoS Hulk", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DoS attacks-SlowHTTPTest", "DoS Slowhttptest", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("DoS attacks-Slowloris", "DoS slowloris", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("FTP-BruteForce", "FTP-Patator", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("Infilteration", "Infiltration", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("SQL Injection", "Web Sql Injection", regex=True) #specified column(s)
data['Label'] = data['Label'].replace("SSH-Bruteforce", "SSH-Patator", regex=True) #specified column(s)

data = data.replace([np.inf, -np.inf], -1) #all columns

# print('renaming: classes') #cicids2017
# data['Label'] = data['Label'].replace(["BENIGN"], "Benign") #specified column(s)
# data['Label'] = data['Label'].replace("Web Attack ", "Web", regex=True) #specified column(s)

print('retyping: force to integer')
data = data.loc[:, data.columns].astype(dtype=int, errors='ignore')
data = data.apply(pd.to_numeric, downcast='integer', errors='ignore')


print('reclassing =',reclass)
if reclass == 1:
    print('Aggregating Data...')
    data['imbalance_label'] = 'Unclassed'  # add column to label grouped classes e.g DDoS
    # data = aggregate_net_class2017(data)
    # data = add_binary_net_class(data)
    # data = resample_net(data)
else:
    print('Concat Only...')

data = get_dups(data)

print("Total rows, cols in output file:", data.shape)
data.to_csv(dir_path + out, index=False, compression='infer')
#
print('saved to', dir_path + out)
