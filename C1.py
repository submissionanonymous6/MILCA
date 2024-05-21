from sklearn.metrics import roc_auc_score
from utils_for_c_tests import *
import time

def train_milca1(train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, p_cutoff=0.01):
    # Compute difference for all features
    train_aucs = []
    betas =np.concatenate((-np.logspace(-6.0, 6.0, num=60),np.logspace(-6.0, 6.0, num=60)), axis=0)
    
    train_predicted_scores_pos = np.zeros((len(train_samples),1))
    train_predicted_scores_neg = np.zeros((len(train_samples),1))
    test_predicted_scores_pos =  np.zeros((len(test_samples),1))
    test_predicted_scores_neg =  np.zeros((len(test_samples),1))

    start_time = time.time()
    # Find significant features
    stat_train=Feature_props(g_train_pos,g_train_neg)
    condition = stat_train[:,1] < p_cutoff # find significant features.
    top_k_index = {}
    for i in range(len(g_train_pos[0][0])):
        if condition[i]:
            top_k_index[i]=np.sign(stat_train[i][1]-stat_train[i][0])
    # Compute difference for significant features
    for i,bag in enumerate(train_samples):
        z_p_tr,z_n_tr=get_two_scores_count(bag, top_k_index)
        train_predicted_scores_pos[i]=z_p_tr
        train_predicted_scores_neg[i]=z_n_tr
    for i,bag in enumerate(test_samples):
        z_p_te,z_n_te=get_two_scores_count(bag, top_k_index)
        test_predicted_scores_pos[i]=z_p_te
        test_predicted_scores_neg[i]=z_n_te

    # in C1 you only use one dataset, so I use two flags to decide if I should use the positive or negative (I use the larger class) and set the other to 0
    beta1,beta2=1,1
    beta2=0
    train_predicted_scores=beta1*train_predicted_scores_pos+beta2*train_predicted_scores_neg
    auc_train=roc_auc_score(train_labels, train_predicted_scores)
    print(auc_train)
    best_beta=1
    if (auc_train<0.5):
        beta1=-beta1
        beta2=-beta2

    test_predicted_scores=beta1*test_predicted_scores_pos-beta2*best_beta*test_predicted_scores_neg
    end_time = time.time()

    # Compute AUC and accuracy, must replace accuracy with best accuracy.
    auc_test=roc_auc_score(test_labels, test_predicted_scores)
    accuracy_tmp = opt_threshold_acc(test_labels, test_predicted_scores)
    accuracy_test=accuracy_tmp[1]
    training_time = end_time - start_time
    return auc_test,accuracy_test, best_beta, training_time