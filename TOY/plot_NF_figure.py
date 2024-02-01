import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from TOY.other_models_for_toy_tests import *
from TOY.data_creator import generate_data
from TRAINERS.C12 import C12_new, train_milca12
from TRAINERS.C1 import train_milca1
from TRAINERS.C2 import train_milca2

def create_NF_figure():
    a = 1
    num_features = [int(x.round()) for x in np.linspace(1, 200, 10)]
    C1_auc = []
    C1_std = []
    C2_auc = []
    C2_std = []
    FC_auc = []
    FC_std = []
    XGB_auc = []
    XGB_std = []
    LR_auc = []
    LR_std = []
    font = fm.FontProperties(family='Times New Roman', size=14)

    # Create a figure and an Axes object
    fig, ax = plt.subplots()

    for nf in tqdm(num_features):
        current_auc_C1 = []
        current_auc_C2 = []
        current_auc_FC = []
        current_auc_XGB = []
        current_auc_LR = []
        for _ in range(10):
            datasetA, datasetB = generate_data(nf, a)
            labels = [1 for b in datasetA] + [0 for b in datasetB]
            data = datasetA + datasetB
            train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                                      random_state=42)
            auc, acc, beta = train_milca1(train_samples, test_samples, train_labels, test_labels, datasetA, datasetB, 0.05)
            current_auc_C1.append(auc)
            auc, acc, beta = train_milca2(train_samples, test_samples, train_labels, test_labels, datasetA, datasetB, 0.05)
            current_auc_C2.append(auc)

            data = stupid_bag_embed(datasetA+datasetB)

            current_auc_FC.append(FC(data, labels))
            current_auc_XGB.append(XGB(data, labels))
            current_auc_LR.append(LR(data, labels))


        C1_auc.append(np.mean(current_auc_C1))
        C1_std.append(np.std(current_auc_C1))
        C2_auc.append(np.mean(current_auc_C2))
        C2_std.append(np.std(current_auc_C2))
        FC_auc.append(np.mean(current_auc_FC))
        FC_std.append(np.std(current_auc_FC))
        XGB_auc.append(np.mean(current_auc_XGB))
        XGB_std.append(np.std(current_auc_XGB))
        LR_auc.append(np.mean(current_auc_LR))
        LR_std.append(np.std(current_auc_LR))


    ax.plot(num_features,FC_auc, label='Fully Connected', color='#4ABD42')
    ax.plot(num_features,C1_auc, label='C1', color='blue')
    ax.plot(num_features,C2_auc, label='C2', color='red')
    ax.plot(num_features,XGB_auc, label='XGB', color='purple')
    ax.plot(num_features,LR_auc, label='Logistic Regression', color='#FF6000')
    ax.plot([0, 200], [0.5, 0.5], 'k--', label='Random')
    ax.set_xlim([10, 200])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Number Of Features', fontproperties=font)
    ax.set_ylabel('AUC', fontproperties=font)
    ax.legend(loc='lower right', prop=font)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("../OUTPUTS/final_AUC_vs_feat_num_plot.pdf", format="pdf")
    plt.show()