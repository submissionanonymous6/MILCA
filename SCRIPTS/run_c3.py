from UTILS.config import model_params, optimization_params
from UTILS.utils_for_c_tests import get_train_test_data, Feature_props, bag_to_onehot
from TRAINERS.C3 import train_milca3
import torch
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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
def train_C3(run_type):
    # Initialize dataset for run.
    datasets_for_run = ['musk1', 'musk2', 'tiger', 'fox', 'elephant', 'v1', 'v2', 'v3', 'v4']

    # Clean the output file.
    output_file = '../OUTPUTS/MILCA_C3_Test_Results.txt' if run_type == 'test' else '../OUTPUTS/MILCA_C3_Training_Results.txt'
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
                        for rs in range(95, 105):
                            torch.manual_seed(rs)
                            np.random.seed(rs)
                            random.seed(rs)

                            # Get the dataset samples
                            train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_data(dataset_name, rs)
                            # Clean the samples + If the run type is 'train' this place the Val samples in the
                            # 'test_samples' variable.
                            train_samples, test_samples, train_labels, test_labels, patience = clean_data_for_C3(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, dataset_name, p, run_type)

                            # Train/Test model
                            auc, acc = train_milca3(train_samples, test_samples, train_labels, test_labels, params, dataset_name, patience, rs)
                            test_aucs.append(auc)
                            test_accs.append(acc)

                        # Save best result
                        mean_auc = np.mean(test_aucs).round(3)
                        mean_acc = np.mean(test_accs).round(3)

                        if mean_auc > best_auc:
                            best_auc = mean_auc
                            mean_auc_str = f"{mean_auc*100}±{(np.std(test_aucs, ddof=1)/np.sqrt(np.size(test_aucs)).round(3))*100}"
                            mean_acc_str = f"{mean_acc*100}±{(np.std(test_aucs, ddof=1)/np.sqrt(np.size(test_aucs)).round(3))*100}"
                            best_params['lr'] = lr
                            best_params['bs'] = bs
                            best_params['p_cutoff'] = p
                            best_params['wd'] = wd

        if run_type == 'test':
            working_file = open(output_file, 'a')
            working_file.write(f"{dataset_name.upper()}\n")
            working_file.write(f'Test AUC mean: {mean_auc_str}\n')
            working_file.write(f'Test Accuracy mean: {mean_acc_str}\n')
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
