import numpy as np
import math
from matplotlib import pyplot as plt
from UTILS.utils_for_c_tests import get_train_test_data, Feature_props



def fig1():
    datasets_for_run = ['musk1', 'musk2', 'fox', 'tiger', 'elephant','v2','v3', 'v4']
    rs=95
    #graphic definitions
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, ax = plt.subplots(2,4, figsize=(7,4))

    for i,dataset_name in enumerate(datasets_for_run):
        left, bottom=0,0
        i1=math.floor(i/4)
        i2=i%4
        if i2==0:
            left=1
        if i1==2:
            bottom=1
        print (i1,i2,bottom,dataset_name)
        train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg = get_train_test_data(dataset_name, rs)
        if dataset_name[0] == 'v':
            dataset_name = f'Web {dataset_name[1]}'
        if 'musk' in dataset_name:
            dataset_name = f'Musk {dataset_name[-1]}'
        plot_scatter(dataset_name, rs,ax[i1][i2], train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg, p_cutof=0.05, left=left, bottom=bottom)
    fig.show()
    fig.savefig('../OUTPUTS/Datasets_distribution.pdf')


def plot_scatter(dataset_name, rs,axis,train_samples, test_samples, train_labels, test_labels, g_train_pos, g_train_neg, g_test_pos, g_test_neg,p_cutof=0.05,left=0,bottom=0):
    size=5
    # Get data
    stat_train=Feature_props(g_train_pos,g_train_neg)
    stat_test=Feature_props(g_test_pos,g_test_neg)
    condition = stat_train[:,1]<p_cutof
    condition1= np.where((stat_train[:,1]<p_cutof) & (stat_train[:,2]<0))[0]
    m=max(max(abs(stat_train[:,2])),max(abs(stat_test[:,2])))

    # plot normalized scatter
    axis.scatter(stat_train[condition,2]/m,stat_test[condition,2]/m,color='red',s=size)
    axis.scatter(stat_train[~condition,2]/m,stat_test[~condition,2]/m,color='green',s=size)
    axis.scatter(stat_train[condition1,2]/m,stat_test[condition1,2]/m,color='blue',s=size)
    freqs,bins=np.histogram(stat_train[:,2], bins=100, range=(-m,m))

    # plot cumsum of training
    freqs=freqs/freqs.sum()
    bins=(bins[1:]+bins[:-1])/2
    axis.plot(bins/m,np.cumsum(freqs),linewidth=2,color='black')

    axis.set_xlim(left=-1.05, right=1.05)
    axis.set_ylim(bottom=-1.05, top=1.05)

    axis.axhline(y=0, color='k')
    axis.axvline(x=0, color='k')
    axis.title.set_text(dataset_name.capitalize())
    if left==1:
        axis.set_ylabel('Test')
    else:
        axis.set_yticklabels([])
    if bottom==1:
        axis.set_xlabel('Training')
    else:
        axis.set_xticklabels([])

if __name__ == '__main__':
    fig1()