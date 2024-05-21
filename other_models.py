import numpy as np
from UTILS.utils_for_c_tests import Feature_props, bag_to_onehot
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.python.keras.callbacks import EarlyStopping
def train_logistic_model(train_samples, test_samples, train_labels, test_labels, ridge=False):
    if ridge:
        logistic_model = LogisticRegression(penalty='l2')
    else:
        logistic_model = LogisticRegression()

    logistic_model.fit(train_samples, train_labels)

    predictions = logistic_model.predict(test_samples)

    accuracy = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, logistic_model.predict_proba(test_samples)[:, 1])

    return accuracy, auc


def train_fully_connected(train_samples, test_samples, train_labels, test_labels):
    # Convert lists to NumPy arrays
    train_samples = np.array(train_samples)
    test_samples = np.array(test_samples)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(train_samples.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_samples, train_labels, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Make predictions
    predictions = model.predict(test_samples)
    predictions_binary = (predictions > 0.5).astype(int)

    # Evaluate performance
    accuracy = accuracy_score(test_labels, predictions_binary)
    auc = roc_auc_score(test_labels, predictions)

    return accuracy, auc

def train_other_method(train_samples, test_samples, train_labels, test_labels ,g_train_pos, g_train_neg,p_cutof, model_name,dataset_name):
    # Find significant features
    stat_train = Feature_props(g_train_pos, g_train_neg)
    condition = stat_train[:, 1] < p_cutof  # find significant features.
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

    if model_name == 'logistic':
        auc_test, accuracy_test = train_logistic_model(train_samples, test_samples, train_labels, test_labels, False)
    elif model_name == 'logistic+ridge':
        auc_test, accuracy_test = train_logistic_model(train_samples, test_samples, train_labels, test_labels, True)
    elif model_name == 'fully_connected':
        auc_test, accuracy_test = train_fully_connected(train_samples, test_samples, train_labels, test_labels)
    return auc_test, accuracy_test
