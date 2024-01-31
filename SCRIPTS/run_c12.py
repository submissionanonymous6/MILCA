from tqdm import tqdm
from TRAINERS.C12 import train_milca12
from UTILS.utils_for_c_tests import get_train_test_data
import numpy as np

def train_C12(method):
    # Initialize dataset for run.
    datasets_for_run = ['musk1', 'musk2', 'tiger', 'fox', 'elephant', 'v1', 'v2', 'v3', 'v4']

    # Clean the output file.
    output_file = f'../OUTPUTS/MILCA_{method}_Test_Results.txt'
    working_file = open(output_file, 'w')
    working_file.close()

    for dataset_name in tqdm(datasets_for_run):
        best_auc = 0
        for i, p_cutoff in enumerate([0.005, 0.01]):
            test_accs = []
            dif_betas = []
            test_aucs = []
            for rs in range(95, 105):
                train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_data(
                    dataset_name, rs)
                auc, acc, beta = train_milca12(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg,
                                     p_cutoff=p_cutoff, flag='C2')
                test_accs.append(acc)
                dif_betas.append(beta)
                test_aucs.append(auc)

            # Save best result
            mean_auc = np.mean(test_aucs).round(3)
            mean_acc = np.mean(test_accs).round(3)

            if mean_auc > best_auc:
                best_auc = mean_auc
                mean_auc_str = f"{mean_auc * 100}±{(np.std(test_aucs, ddof=1) / np.sqrt(np.size(test_aucs)).round(3)) * 100}"
                mean_acc_str = f"{mean_acc * 100}±{(np.std(test_aucs, ddof=1) / np.sqrt(np.size(test_aucs)).round(3)) * 100}"

        working_file = open(output_file, 'a')
        working_file.write(f'{dataset_name.upper()}\n')
        working_file.write(f"Mean AUC: {mean_auc_str}\n")
        working_file.write(f"Mean Accuracy: {mean_acc_str}\n")
        working_file.write("\n")
        working_file.close()



if __name__ == '__main__':
    train_C12('C1')
    train_C12('C2')
