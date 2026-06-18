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

X = np.load("datasets/X_gear.npy")
y = np.load("datasets/y_gear.npy")
groups = np.load("datasets/groups_gear.npy")

print(pd.Series(y).value_counts().sort_index())
print(X.shape)

# Encode gear labels: strings -> integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_
num_classes = len(class_names)

print("Classes:", class_names)
print("Encoded labels:", dict(zip(class_names, range(num_classes))))

# First split: train / temp
sgkf1 = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

train_idx, temp_idx = next(
    sgkf1.split(X, y_encoded, groups=groups)
)

X_train, y_train = X[train_idx], y_encoded[train_idx]
groups_train = groups[train_idx]

X_temp, y_temp = X[temp_idx], y_encoded[temp_idx]
groups_temp = groups[temp_idx]

# Second split: validation / test
sgkf2 = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=2
)

val_idx, test_idx = next(
    sgkf2.split(X_temp, y_temp, groups=groups_temp)
)

X_val, y_val = X_temp[val_idx], y_temp[val_idx]
X_test, y_test = X_temp[test_idx], y_temp[test_idx]

groups_val = groups_temp[val_idx]
groups_test = groups_temp[test_idx]

print("Full dataset:")
print(pd.Series(y).value_counts())

print("\nTrain:")
print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())

print("\nValidation:")
print(pd.Series(label_encoder.inverse_transform(y_val)).value_counts())

print("\nTest:")
print(pd.Series(label_encoder.inverse_transform(y_test)).value_counts())

groups_val = groups_temp[val_idx]
groups_test = groups_temp[test_idx]

print("Groups:")
print("Train groups:", len(np.unique(groups_train)))
print("Val groups:", len(np.unique(groups_val)))
print("Test groups:", len(np.unique(groups_test)))

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_val = scaler.transform(
    X_val.reshape(-1, X_val.shape[-1])
).reshape(X_val.shape)

X_test = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


tf.keras.backend.clear_session()

model = build_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    num_classes=num_classes
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

model.save("gear_cnn.keras")

# Save the preprocessing objects
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("Accuracy:", acc)
print("Macro precision:", prec)
print("Macro recall:", rec)
print("Macro F1:", f1)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4
))


