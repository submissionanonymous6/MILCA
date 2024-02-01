import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
# from UTILS.utils_for_c_tests import *
from TOY.other_models_for_toy_tests import *
from TOY.data_creator import generate_data
from TRAINERS.C12 import C12_new, train_milca12
from TRAINERS.C1 import train_milca1
from TRAINERS.C2 import train_milca2

def create_alpha_figure():
    alpha = np.logspace(np.log10(10**-2), np.log10(1), 10)
    num_features = 150
    C1_auc = []
    C2_auc = []
    FC_auc = []
    XGB_auc = []
    LR_auc = []

    for a in tqdm(alpha):
        current_auc_C1 = []
        current_auc_C2 = []
        current_auc_FC = []
        current_auc_XGB = []
        current_auc_LR = []
        for _ in range(10):
            datasetA, datasetB = generate_data(num_features, a)
            labels = [1 for b in datasetA] + [0 for b in datasetB]
            data = datasetA+datasetB

            train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
            auc, acc, beta= train_milca1(train_samples, test_samples, train_labels, test_labels, datasetA, datasetB, 0.05,'C1')
            current_auc_C1.append(auc)
            auc, acc, beta = train_milca2(train_samples, test_samples, train_labels, test_labels, datasetA, datasetB, 0.05,'C2')
            current_auc_C2.append(auc)

            data = stupid_bag_embed(datasetA+datasetB)

            current_auc_FC.append(FC(data, labels))
            # current_auc_FC.append(0.1)
            current_auc_XGB.append(XGB(data, labels))
            # current_auc_XGB.append(0.2)
            current_auc_LR.append(LR(data, labels))
            # current_auc_LR.append(0.3)


        C1_auc.append(np.mean(current_auc_C1))
        C2_auc.append(np.mean(current_auc_C2))
        FC_auc.append(np.mean(current_auc_FC))
        XGB_auc.append(np.mean(current_auc_XGB))
        LR_auc.append(np.mean(current_auc_LR))

    font = fm.FontProperties(family='Times New Roman', size=14)

    # Create a figure and an Axes object
    fig, ax = plt.subplots()
    ax.plot(alpha,FC_auc, label='Fully Connected', color='#4ABD42')
    ax.plot(alpha,XGB_auc, label='XGB', color='purple')
    ax.plot(alpha,LR_auc, label='Logistic Regression', color='#FF6000')
    ax.plot(alpha,C1_auc, label='C1', color='blue')
    ax.plot(alpha,C2_auc, label='C2', color='red')


    ax.plot([0, 200], [0.5, 0.5], 'k--', label='Random')
    ax.set_xscale('log')
    ax.set_xlim([10**-2, 1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(r'$\alpha$', fontproperties=font)  # Use LaTeX formatting for alpha
    ax.set_ylabel('AUC', fontproperties=font)
    ax.legend(loc='lower right', prop=font)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("../OUTPUTS/final_AUC_vs_Alpha_plot.pdf", format="pdf")
    plt.show()
