import numpy as np
import scipy.stats as stats
from os.path import exists
import pickle


# This function returns the significant features under the C3 restrictions.
def top_k_for_C3(datasetA, datasetB):
    best_feat = []
    for i in range(len(datasetA[0][0])):
        dist_i_A, dist_i_B = get_dist(datasetA, i), get_dist(datasetB, i)

        statistic, p_value = stats.mannwhitneyu(dist_i_A, dist_i_B)

        if p_value < 0.05:
            best_feat.append(i)
    return best_feat

# This function gets a dataset and an index, and returns the distribution(list of means) of the index in the bags.
def get_dist(dataset, index):
    dist = []
    for bag in dataset:
        feat_count = 0
        for instance in bag:
            feat_count += instance[index]
        dist.append(feat_count / len(bag))
    return dist

# This function gets a bag and a list of features and returns a vector of means.
def bag_to_onehot(bag, top_k):
    num_instances = len(bag)
    num_words = len(top_k)
    onehot = [0] * num_words

    for instance in bag:
        for j, index in enumerate(top_k):
            onehot[j] += instance[index]

    return [x/num_instances for x in onehot]


def Feature_props(datasetA, datasetB):
    N = len(datasetA[0][0])
    statistic = np.zeros((N, 3))
    for i in range(N):
        dist_i_A, dist_i_B = get_dist(datasetA, i), get_dist(datasetB, i)
        statistic[i, 0], statistic[i, 1] = stats.ttest_ind(dist_i_A, dist_i_B)
        statistic[i, 2] = np.mean(dist_i_A) - np.mean(dist_i_B)
    return statistic


def get_two_scores_count(bag, best_feat):
    inst_count, feat_count_p, feat_count_n = len(bag), 0, 0
    for instance in bag:
        for i in range(len(instance)):
            if i in best_feat.keys():
                if best_feat[i] == 1:
                    feat_count_p += instance[i]
                else:
                    feat_count_n += instance[i]
    return feat_count_p / inst_count, feat_count_n / inst_count


def stupid_bag_embed(data):
    embedded_data = []
    for bag in data:
        bag = np.array(bag)
        embedded_data.append(np.mean(bag,axis=0))
    return embedded_data

def opt_threshold_acc(y_true, y_pred):
    A = list(zip(y_true, y_pred))
    A = sorted(A, key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0] == 1])
    tn = 0
    th_acc = []
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        th_acc.append((th, acc))
    return max(th_acc, key=lambda x: x[1])


def get_train_test_data(dataset_name, rs):
    full_name='DATA/'+dataset_name+'_'+str(rs)+'.pkl'
    if exists(full_name):
        with open(full_name,'rb') as f:  # Python 3: open(..., 'rb')
            train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg = pickle.load(f)
        # Those are the positive and negative groups in the training and test.
        g_train_pos=[x for i, x in enumerate(train_samples) if train_labels[i] == 1]
        g_train_neg=[x for i, x in enumerate(train_samples) if train_labels[i] == (0 if 'musk' in dataset_name else -1)]
        g_test_pos=[x for i, x in enumerate(test_samples) if test_labels[i] == 1]
        g_test_neg=[x for i, x in enumerate(test_samples) if test_labels[i] == (0 if 'musk' in dataset_name else -1)]
        with open(full_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg], f)

    # Getting back the objects:
    return train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg
