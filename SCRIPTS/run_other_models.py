from TRAINERS.other_models import train_other_method
import numpy as np
from tqdm import tqdm
from UTILS.utils_for_c_tests import get_train_test_data
def train_other_models():
    # Initiate Runs:
    datasets_for_run = ['musk1', 'musk2', 'tiger', 'fox', 'elephant', 'v1', 'v2', 'v3', 'v4']
    models_for_run = ['logistic', 'logistic+ridge','fully_connected']

    for model_name in tqdm(models_for_run):
        output_file = f'../OUTPUTS/Results_{model_name}.txt'
        working_file = open(output_file, 'w')
        working_file.write(f'{model_name.upper()}\n')
        working_file.close()
        for dataset_name in datasets_for_run:
            best_auc = 0
            for i, p_cutoff in enumerate([0.005, 0.01]):
                test_accs = []
                dif_betas = []
                test_aucs = []
                for rs in range(95, 105):
                    train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_negs = get_train_test_data(
                        dataset_name, rs)
                    auc, acc = train_other_method(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg,p_cutoff, model_name,dataset_name)
                    test_accs.append(acc)
                    test_aucs.append(auc)

                # Save best result
                mean_auc = np.mean(test_aucs).round(3)
                mean_acc = np.mean(test_accs).round(3)

                if mean_auc > best_auc:
                    best_auc = mean_auc
                    mean_auc_str = f"{mean_auc * 100}±{(np.std(test_aucs).round(3)) * 100}"
                    mean_acc_str = f"{mean_acc * 100}±{(np.std(test_accs).round(3)) * 100}"

            working_file = open(output_file, 'a')
            working_file.write(f'{dataset_name.upper()}\n')
            working_file.write(f"Mean AUC: {mean_auc_str}\n")
            working_file.write(f"Mean Accuracy: {mean_acc_str}\n")
            working_file.write("\n")
            working_file.close()


if __name__ == "__main__":
    train_other_models()


