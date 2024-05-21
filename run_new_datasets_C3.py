
from config import model_params, optimization_params
from utils_for_c_tests import get_train_test_data, Feature_props, bag_to_onehot
from C3 import train_milca3
import torch
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import csv
import pandas as pd
from data_creator import read_synthetic_dataset
from sklearn.model_selection import train_test_split
import time

def get_city_data():
    filename = "DATA/wiki_countries_small.csv"

    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    df = pd.DataFrame(data)

    # get the unique names of the cities
    unique_names = df[0].unique()[1:]

    # create a dictionary with the unique names as keys and empty lists as values
    cities_dict = {name: [] for name in unique_names}

    # iterate over the rows of the dataframe and append the values to the lists in the dictionary, skip the first row
    for index, row in df.iterrows():
        if index == 0:
            continue
        cities_dict[row[0]].append(row[1:])

    # create an empty list to store the tensors
    bags_list = []
    labels = []
    # iterate over each city in the unique_names list
    for city in unique_names:
        # get the list of dataframes for the current city
        city_data = cities_dict[city]
        
        # get the country id from the first element on the first row of the city_data
        country_id = int(list(city_data[0])[0][1])

        instances = []
        for instance in city_data:
            instances.append([float(item) for item in list(instance[1:])])
            
        # create a tensor from the list of lists and append it to the bags_list
        bags_list.append(instances)
        labels.append(country_id)

    # make the labels binary (1 if the label is 4, 0 otherwise)
    labels = [1 if label == 4 else 0 for label in labels]

    return bags_list, labels

def get_train_test_data_wiki(rs):
    bags_list, labels = get_city_data()
    # split to train and test using sci-kit learn
    train_samples, test_samples, train_labels, test_labels = train_test_split(bags_list, labels, test_size=0.2, random_state=rs)
    
    g_train_pos = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 1]
    g_train_neg = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 0]
    g_test_pos = [test_samples[i] for i in range(len(test_samples)) if test_labels[i] == 1]
    g_test_negs = [test_samples[i] for i in range(len(test_samples)) if test_labels[i] == 0]

    return train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs

def get_train_test_data_synth(rs):
    data, labels = read_synthetic_dataset()
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=rs)

    g_train_pos = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 1]
    g_train_neg = [train_samples[i] for i in range(len(train_samples)) if train_labels[i] == 0]
    g_test_pos = [test_samples[i] for i in range(len(test_samples)) if test_labels[i] == 1]
    g_test_negs = [test_samples[i] for i in range(len(test_samples)) if test_labels[i] == 0]

    return train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs

def get_train_test_new_datasets(name, rs):
    if name == 'wiki':
        return get_train_test_data_wiki(rs)
    else:
        return get_train_test_data_synth(rs)
    
def clean_data_for_C3(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, dataset_name, p_cutoff, run_type):
    start_time = time.time()
    stat_train = Feature_props(g_train_pos, g_train_neg)
    condition = stat_train[:, 1] < p_cutoff  # find significant features.
    top_k_index = {}

    for i in range(len(g_train_pos[0][0])):
        if condition[i]:
            top_k_index[i] = np.sign(stat_train[i][1] - stat_train[i][0])

    if dataset_name[0] != 'v':
        # Encode bags
        train_samples = [bag_to_onehot(bag, top_k_index) for bag in train_samples]
        test_samples = [bag_to_onehot(bag, top_k_index) for bag in test_samples]

    else:
        # Fix shape
        train_samples = [bag[0] for bag in train_samples]
        test_samples = [bag[0] for bag in test_samples]

    # Fix Labels
    train_labels = [0 if label == -1 else label for label in train_labels]
    test_labels = [0 if label == -1 else label for label in test_labels]

    # Normalize data
    scaler = MinMaxScaler().fit(train_samples)
    train_samples = scaler.transform(train_samples)
    test_samples = scaler.transform(test_samples)
    end_time = time.time()
    if run_type != 'test':
        validation_index = int(len(train_samples) * 0.8)
        train_samples, test_samples = train_samples[:validation_index], train_samples[validation_index:]
        train_labels, test_labels = train_labels[:validation_index], train_labels[validation_index:]

    patience = 100 if run_type == 'test' else 25

    processing_time = end_time - start_time

    return train_samples, test_samples, train_labels, test_labels, patience, processing_time

def train_C3(run_type):
    # Initialize dataset for run.
    datasets_for_run = ['wiki', 'synth']

    # Clean the output file.
    output_file = 'MILCA_C3_Wiki_Test_Results.txt'
    working_file = open(output_file, 'w')
    working_file.close()


    # Train/Test all datasets.
    for dataset_name in tqdm(datasets_for_run):
        # Check type of run - test will return dic of optimal parameters, train will return dic of optional values.
        config = model_params[dataset_name] if run_type == 'test' else optimization_params

        best_params = {}
        best_auc = 0

        # If run_type == 'test' then this is just a single run.
        for lr in config['lr']:
            for wd in config['wd']:
                for bs in config['bs']:
                    for p in config['p_cutoff']:
                        params = {'lr': lr, 'wd': wd, 'bs': bs}
                        test_aucs = []
                        test_accs = []
                        times = []
                        for rs in range(95, 105):
                            torch.manual_seed(rs)
                            np.random.seed(rs)
                            random.seed(rs)

                            # Get the dataset samples
                            train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_new_datasets(dataset_name, rs)
                            # Clean the samples + If the run type is 'train' this place the Val samples in the
                            # 'test_samples' variable.
                            train_samples, test_samples, train_labels, test_labels, patience, processing_time = clean_data_for_C3(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, dataset_name, p, run_type)

                            # Train/Test model
                            auc, acc, running_time = train_milca3(train_samples, test_samples, train_labels, test_labels, params, dataset_name, patience, rs)
                            test_aucs.append(auc)
                            test_accs.append(acc)
                            times.append(running_time+processing_time)

                        # Save best result
                        mean_auc = np.mean(test_aucs).round(3)
                        mean_acc = np.mean(test_accs).round(3)
                        mean_time = np.mean(times).round(3)

                        if mean_auc > best_auc:
                            best_auc = mean_auc
                            mean_auc_str = f"{mean_auc*100}±{(np.std(test_aucs, ddof=1)/np.sqrt(np.size(test_aucs)).round(3))*100}"
                            mean_acc_str = f"{mean_acc*100}±{(np.std(test_aucs, ddof=1)/np.sqrt(np.size(test_aucs)).round(3))*100}"
                            mean_time_str = f"{mean_time}±{(np.std(times, ddof=1)/np.sqrt(np.size(times)).round(3))}"
                            best_params['lr'] = lr
                            best_params['bs'] = bs
                            best_params['p_cutoff'] = p
                            best_params['wd'] = wd

        if run_type == 'test':
            working_file = open(output_file, 'a')
            working_file.write(f"{dataset_name.upper()}\n")
            working_file.write(f'Test Accuracy mean: {mean_acc_str}\n')
            working_file.write(f'Test AUC mean: {mean_auc_str}\n')
            working_file.write(f'training Time mean: {mean_time_str}\n')
            working_file.write(f'With LR: {best_params["lr"]}\n')
            working_file.write(f'And WD: {best_params["wd"]}\n')
            working_file.write(f'And BS: {best_params["bs"]}\n')
            working_file.write(f'And cutoff: {best_params["p_cutoff"]}\n')
            working_file.write("\n\n")
            working_file.close()
        else:
            working_file = open(output_file, 'a')
            working_file.write(f"{dataset_name.upper()}\n")
            working_file.write(f'Val AUC mean: {mean_auc_str}\n')
            working_file.write(f'Val Accuracy mean: {mean_acc_str}\n')
            working_file.write(f'With LR: {best_params["lr"]}\n')
            working_file.write(f'And WD: {best_params["wd"]}\n')
            working_file.write(f'And BS: {best_params["bs"]}\n')
            working_file.write(f'And cutoff: {best_params["p_cutoff"]}\n')
            working_file.write("\n\n")
            working_file.close()


if __name__ == '__main__':
    # Un-comment test if you want to run the model on the optimal hyperparameters.
    # Un-comment train if you want to optimize the model.
    train_C3('test')
    # train_C3('train')