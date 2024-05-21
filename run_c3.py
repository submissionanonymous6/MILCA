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
from sklearn.model_selection import train_test_split
import time
import warnings


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


def clean_data_for_C3(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, dataset_name, p_cutoff, run_type):
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

    if run_type != 'test':
        validation_index = int(len(train_samples) * 0.8)
        train_samples, test_samples = train_samples[:validation_index], train_samples[validation_index:]
        train_labels, test_labels = train_labels[:validation_index], train_labels[validation_index:]

    patience = 100 if run_type == 'test' else 25

    return train_samples, test_samples, train_labels, test_labels, patience

def train_C3(run_type, hyper_parameters=None, dataset_name=None):
    # Initialize a file for final results.
    output_file = f'MILCA_C3_Final_Results.txt'
    working_file = open(output_file, 'w')
    working_file.close()

    total_datasets_mean_acc = []

    # Initialize dataset for run.
    # datasets_for_run = ['musk1', 'musk2', 'fox', 'tiger', 'elephant', 'v1', 'v2', 'v3', 'v4','wiki'] if run_type == 'train' else [dataset_name]
    datasets_for_run = ['wiki'] if run_type == 'train' else [dataset_name]

    # Train/Test all datasets.
    for dataset_name in datasets_for_run:

        # Clean the output file.
        output_file = f'MILCA_C3_{dataset_name}_Test_Results.txt' if run_type == 'test' else f'MILCA_C3_{dataset_name}_Training_Results.txt'
        working_file = open(output_file, 'w')
        working_file.close()

        # Check type of run - test will return dic of optimal parameters, train will return dic of optional values.
        hyper_parameters = [] if run_type == "train" else hyper_parameters
        seed_aucs = []
        seed_accs = []
        seed_times = []  

        for rs in tqdm(range(95, 105), desc=f'{run_type.capitalize()} - {dataset_name}', leave=True):
            config = hyper_parameters[rs-95] if run_type == 'test' else optimization_params
            
            best_auc = 0

            torch.manual_seed(rs)
            np.random.seed(rs)
            random.seed(rs)
            
            # Get the dataset samples
            if dataset_name == "wiki":
                train_samples_og, test_samples_og, train_labels_og, test_labels_og, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_data_wiki(rs)
            else:
                train_samples_og, test_samples_og, train_labels_og, test_labels_og, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_data(dataset_name, rs)
                    
            # If run_type == 'test' then this is just a single run.
            for lr in config['lr']:
                for wd in config['wd']:
                    for bs in config['bs']:
                        for p in config['p_cutoff']:
                            params = {'lr': lr, 'wd': wd, 'bs': bs}

                            # Copy the samples to avoid changing the original data.
                            test_samples = test_samples_og.copy()
                            test_labels = test_labels_og.copy()
                            train_samples = train_samples_og.copy()
                            train_labels = train_labels_og.copy()

                            # Clean the samples + If the run type is 'train' this place the Val samples in the
                            # 'test_samples' variable.
                            train_samples, test_samples, train_labels, test_labels, patience = clean_data_for_C3(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, dataset_name, p, run_type)

                            # Train/Test model
                            auc, acc, running_time = train_milca3(train_samples, test_samples, train_labels, test_labels, params, dataset_name, patience, rs)

                            # If run_type == 'train' then we are optimizing the model.
                            if run_type == 'train':   
                                if auc > best_auc:
                                    best_auc = auc
                                    best_acc = acc
                                    best_time = running_time
                                    best_hyperparameters = {'lr': [lr], 'wd': [wd], 'bs': [bs], 'p_cutoff': [p]}

                            else:
                                best_auc = auc
                                best_acc = acc
                                best_time = running_time

            seed_aucs.append(best_auc)
            seed_accs.append(best_acc)
            seed_times.append(best_time)

            if run_type == 'train':
                hyper_parameters.append(best_hyperparameters)

        
        # notice that the values in the seed_lists are tuples.

        # print(seed_aucs)
        mean_auc, sem_auc = round(np.mean(seed_aucs) * 100, 1), round(np.std(seed_aucs, ddof=1) / np.sqrt(np.size(seed_aucs)) * 100, 1)
        mean_acc, sem_acc = round(np.mean(seed_accs) * 100, 1), round(np.std(seed_accs, ddof=1) / np.sqrt(np.size(seed_accs)) * 100, 1)
        mean_time, sem_time = round(np.mean(seed_times), 1), round(np.std(seed_times, ddof=1) / np.sqrt(np.size(seed_times)), 1)

        if run_type == 'test':
            working_file = open(output_file, 'a')
            working_file.write(f"{dataset_name.upper()}\n")
            working_file.write(f'Test AUC mean: {mean_auc}±{sem_auc}\n')
            working_file.write(f'Test Accuracy mean: {mean_acc}±{sem_acc}\n')
            working_file.write(f'The different accuracys were: {seed_accs}\n')
            working_file.write(f'Test Time mean: {mean_time}±{sem_time}\n')
            working_file.write("\n\n")
            working_file.close()
            return(mean_acc)
        else:
            working_file = open(output_file, 'a')
            working_file.write(f"{dataset_name.upper()}\n")
            working_file.write(f'Val AUC mean: {mean_auc}±{sem_auc}\n')
            working_file.write(f'Val Accuracy mean: {mean_acc}±{sem_acc}\n')
            working_file.write(f'The different accuracys were: {seed_accs}\n')
            working_file.write(f'Val Time mean: {mean_time}±{sem_time}\n')
            working_file.write("\n\n")
            working_file.close()

            total_datasets_mean_acc.append(train_C3('test', hyper_parameters, dataset_name))
    
    working_file = open(output_file, 'a')
    working_file.write(f'Total Test mean accuracy: {round(np.mean(total_datasets_mean_acc), 1)}\n')
    working_file.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_C3('train')
    
