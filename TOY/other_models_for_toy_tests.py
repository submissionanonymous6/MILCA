from sklearn.model_selection import train_test_split
import xgboost as xgb
from UTILS.utils_for_c_tests import *
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.python.keras.callbacks import EarlyStopping


def FC(data, labels):
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              random_state=42)

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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(train_samples, train_labels, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Make predictions
    predictions = model.predict(test_samples)

    # Evaluate performance
    auc = roc_auc_score(test_labels, predictions)

    return auc


def XGB(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score


def LR(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc
