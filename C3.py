import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from utils_for_c_tests import opt_threshold_acc
import time
# Initializing a DataLoaders elements for training
def get_data_loaders(train_samples, test_samples, train_labels, test_labels, dim, batch_size, validation_split=0.2):
    # Split the training data into training and validation
    train_data = list(zip(train_samples, train_labels))
    split_index = int(len(train_data) * validation_split)
    train_data, val_data = train_data[split_index:], train_data[:split_index]
    train_samples, train_labels = zip(*train_data)
    val_samples, val_labels = zip(*val_data)

    # convert to torches with correct shape
    train_samples = torch.from_numpy(np.reshape(np.array(train_samples), (len(train_samples), dim)))
    val_samples = torch.from_numpy(np.reshape(np.array(val_samples), (len(val_samples), dim)))
    test_samples = torch.from_numpy(np.reshape(np.array(test_samples), (len(test_samples), dim)))

    train_labels = torch.from_numpy(np.array(train_labels))
    val_labels = torch.from_numpy(np.array(val_labels))
    test_labels = torch.from_numpy(np.array(test_labels))

    # Create datasets and loaders
    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)
    test_dataset = TensorDataset(test_samples, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MILCA3(nn.Module):
    def __init__(self, dim):
        super(MILCA3, self).__init__()
        self.mu = nn.Parameter(torch.rand(dim))
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.1)

        # Print the number of learnable parameters
        # self._print_num_parameters()

    def forward(self, x):
        alphas = torch.tanh(self.mu)
        x = self.layer1(self.dropout(alphas * x.float()))
        x = self.layer2(self.dropout(x))
        return torch.sigmoid(torch.sum(x, dim=1, keepdim=True)).squeeze(1)

    def _print_num_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([torch.numel(p) for p in model_parameters])
        print(f'Number of learnable parameters in MILCA3 model: {num_params}')


# Training
def train_milca3(train_samples, test_samples, train_labels, test_labels, config, dataset_name, patience, rs):
    epochs = 100
    early_stop_counter = 0

    dim = len(train_samples[0])
    model = MILCA3(dim)
    device = 'cpu'
    criterion = nn.BCELoss()

    # Set hyper-parameters
    batch_size = config['bs']
    lr = config['lr']
    wd = config['wd']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_samples = torch.from_numpy(np.array(train_samples))
    train_labels = torch.from_numpy(np.array(train_labels))
    test_samples = torch.from_numpy(np.array(test_samples))
    test_labels = torch.from_numpy(np.array(test_labels))

    # Create datasets and loaders
    train_dataset = TensorDataset(train_samples, train_labels)
    test_dataset = TensorDataset(test_samples, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    loss_history = {'train': [], 'test': []}

    # Training loop
    best_test_acc = 0
    best_test_auc = 0
    # calculate running time
    running_time = 0
    for epoch in range(epochs):
        training_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        running_test_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
        training_stop_time = time.time()
        running_time += training_stop_time - training_start_time
        average_train_loss = running_train_loss / len(train_loader)
        loss_history['train'].append(average_train_loss)

        test_outputs = []
        test_labels = []
        # Test performance
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs.float(), labels.float())
                running_test_loss += loss.item()

                test_outputs.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        average_test_loss = running_test_loss / len(test_loader)
        loss_history['test'].append(average_test_loss)

        # Calculate AUC
        test_auc = roc_auc_score(test_labels, test_outputs)
        test_accuracy = opt_threshold_acc(test_labels, test_outputs)[1]

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_test_auc = test_auc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            # print(f'Early stopping at epoch {epoch + 1} with AUC: {best_test_auc} and ACC: {best_test_acc}')
            break


    return best_test_auc, best_test_acc, running_time
