import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
import joblib

X_train = np.load("datasets/X_train_all.npy")
y_train = np.load("datasets/y_train_all.npy", allow_pickle=True)
groups_train = np.load("datasets/groups_train_all.npy")

X_val = np.load("datasets/X_val_all.npy")
y_val = np.load("datasets/y_val_all.npy", allow_pickle=True)
groups_val = np.load("datasets/groups_val_all.npy")

X_test_unseen = np.load("datasets/X_test_unseen_all.npy")
y_test_unseen = np.load("datasets/y_test_unseen_all.npy", allow_pickle=True)
groups_test_unseen = np.load("datasets/groups_test_unseen_all.npy")

X_test_seen = np.load("datasets/X_test_seen_all.npy")
y_test_seen = np.load("datasets/y_test_seen_all.npy", allow_pickle=True)
groups_test_seen = np.load("datasets/groups_test_seen_all.npy")

# Encode gear labels: strings -> integers
label_encoder = LabelEncoder().fit(y_train)
class_names = label_encoder.classes_          # ['Garn','Krokredskap','Snurrevad','Trål'] (alphabetical)
num_classes = len(class_names)

y_train        = label_encoder.transform(y_train)
y_val          = label_encoder.transform(y_val)
y_test_unseen  = label_encoder.transform(y_test_unseen)
y_test_seen    = label_encoder.transform(y_test_seen)

print("Classes:", class_names)
print("Encoded labels:", dict(zip(class_names, range(num_classes))))

print("\nTrain:")
print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())

print("\nValidation:")
print(pd.Series(label_encoder.inverse_transform(y_val)).value_counts())

print("\nTest unseen:")
print(pd.Series(label_encoder.inverse_transform(y_test_unseen)).value_counts())

print("\nTest seen:")
print(pd.Series(label_encoder.inverse_transform(y_test_seen)).value_counts())


print("Groups:")
print("Train groups:", len(np.unique(groups_train)))
print("Val groups:", len(np.unique(groups_val)))
print("Test unseen groups:", len(np.unique(groups_test_unseen)))
print("Test seen groups:", len(np.unique(groups_test_seen)))

# Class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight = dict(zip(classes, weights))
print("Class weights:", class_weight)


def build_model(input_shape, num_classes):
    inp = tf.keras.layers.Input(shape=input_shape)          # (120, 10)
    n_features = input_shape[-1] - 1

    feats = tf.keras.layers.Lambda(lambda z: z[:, :, :n_features], name="features")(inp)
    mask  = tf.keras.layers.Lambda(lambda z: z[:, :, n_features:], name="mask")(inp)

    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(feats)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    m = tf.keras.layers.MaxPooling1D(2)(mask)               # 1 if any real step in the pool

    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    m = tf.keras.layers.MaxPooling1D(2)(m)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t[0] * t[1], axis=1) / (tf.reduce_sum(t[1], axis=1) + 1e-6)
    )([x, m])                                               # masked global average

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tf.keras.backend.clear_session()

model = build_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    num_classes=num_classes
)

model.summary()

class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        logs["val_macro_f1"] = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        print(f" — val_macro_f1: {logs['val_macro_f1']:.4f}")

macro_cb = MacroF1(X_val, y_val)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_macro_f1",
    mode="max",                 # higher F1 is better — auto would have assumed min
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[macro_cb, early_stop],   # macro_cb first so the key exists when early_stop reads it
    verbose=2,
)

model.save_weights("gear_cnn_2023_train_weights.h5")
joblib.dump(label_encoder, "gear_label_encoder.joblib")

# Predict UNSEEN
y_prob_unseen = model.predict(X_test_unseen, verbose=0)
y_pred_unseen = np.argmax(y_prob_unseen, axis=1)

# Metrics UNSEEN
print("=======UNSEEN TEST RESULTS=========")
acc = accuracy_score(y_test_unseen, y_pred_unseen)
prec = precision_score(y_test_unseen, y_pred_unseen, average="macro", zero_division=0)
rec = recall_score(y_test_unseen, y_pred_unseen, average="macro", zero_division=0)
f1 = f1_score(y_test_unseen, y_pred_unseen, average="macro", zero_division=0)

print("Accuracy:", acc)
print("Macro precision:", prec)
print("Macro recall:", rec)
print("Macro F1:", f1)

print("\nConfusion matrix:")
print(confusion_matrix(y_test_unseen, y_pred_unseen))

print("\nClassification report:")
print(classification_report(
    y_test_unseen,
    y_pred_unseen,
    target_names=class_names,
    digits=4
))

# Predict SEEN
y_prob_seen = model.predict(X_test_seen, verbose=0)
y_pred_seen = np.argmax(y_prob_seen, axis=1)

# Metrics SEEN
print("=======SEEN TEST RESULTS=========")
acc_seen = accuracy_score(y_test_seen, y_pred_seen)
prec_seen = precision_score(y_test_seen, y_pred_seen, average="macro", zero_division=0)
rec_seen = recall_score(y_test_seen, y_pred_seen, average="macro", zero_division=0)
f1_seen = f1_score(y_test_seen, y_pred_seen, average="macro", zero_division=0)

print("Accuracy:", acc_seen)
print("Macro precision:", prec_seen)
print("Macro recall:", rec_seen)
print("Macro F1:", f1_seen)

print("\nConfusion matrix:")
print(confusion_matrix(y_test_seen, y_pred_seen))

print("\nClassification report:")
print(classification_report(
    y_test_seen,
    y_pred_seen,
    target_names=class_names,
    digits=4
))
