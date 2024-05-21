from tqdm import tqdm
from C1 import train_milca1
from C2 import train_milca2
from run_new_datasets_C3 import get_train_test_new_datasets
from utils_for_c_tests import get_train_test_data
import numpy as np

def train_C12(method):
    # Initialize dataset for run.
    datasets_for_run = ['musk1', 'musk2', 'tiger', 'fox', 'elephant', 'v1', 'v2', 'v3', 'v4']
    myz_datasets = ["wiki","synth"]
    # Clean the output file.
    output_file = f'new_MILCA_{method}_wiki_synth_Test_Results.txt'
    working_file = open(output_file, 'w')
    working_file.close()

    for dataset_name in tqdm(my_datasets):
        best_auc = 0
        for i, p_cutoff in enumerate([0.005, 0.01]):
            test_accs = []
            dif_betas = []
            test_aucs = []
            times = []
            for rs in range(95, 105):
                train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_new_datasets(
                    dataset_name, rs)
                if method == 'C1':
                    auc, acc, beta, training_time = train_milca1(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg,
                                     p_cutoff=p_cutoff)
                elif method == 'C2':
                    auc, acc, beta, training_time = train_milca2(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg,
                                     p_cutoff=p_cutoff)
                test_accs.append(acc)
                dif_betas.append(beta)
                test_aucs.append(auc)
                times.append(training_time)

            # Save best result
            mean_auc = np.mean(test_aucs).round(3)
            mean_acc = np.mean(test_accs).round(3)
            mean_time = np.mean(times).round(3)

            if mean_auc > best_auc:
                best_auc = mean_auc
                mean_auc_str = f"{mean_auc * 100}±{(np.std(test_aucs, ddof=1) / np.sqrt(np.size(test_aucs)).round(3)) * 100}"
                mean_acc_str = f"{mean_acc * 100}±{(np.std(test_aucs, ddof=1) / np.sqrt(np.size(test_aucs)).round(3)) * 100}"
                mean_time_str = f"{mean_time}±{(np.std(times, ddof=1) / np.sqrt(np.size(times)).round(3))}"

        working_file = open(output_file, 'a')
        working_file.write(f'{dataset_name.upper()}\n')
        working_file.write(f"Mean Accuracy: {mean_acc_str}\n")
        working_file.write(f"Mean AUC: {mean_auc_str}\n")
        working_file.write(f"Mean Time: {mean_time_str}\n")
        working_file.write("\n")
        working_file.close()



if __name__ == '__main__':
    # train_C12('C1')
    train_C12('C2')
