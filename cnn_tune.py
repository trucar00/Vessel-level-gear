import optuna, json
import numpy as np, pandas as pd, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Visible GPUs:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Disable XLA; it fails during cuDNN autotuning on this P100.
tf.config.optimizer.set_jit(False)

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

GEARS = ["Trål", "Krokredskap", "Not", "Snurrevad", "Garn"]
BASE_FEATURES   = ["cog_interp_sin", "cog_interp_cos", "speed_calc_ms",
                   "ra_accel", "ra_jerk", "log_dist", "ra_dcog"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES

FEATS_PATH = "../../LSTM/three_months/resampled"
TUNING_FILES = [f"{FEATS_PATH}/2023_1_3_feats.parquet", f"{FEATS_PATH}/2023_7_9_feats.parquet"]          # add the other quarters

CACHE_DIR = Path("win_cache"); CACHE_DIR.mkdir(exist_ok=True)
WINDOWS     = [60, 120, 240]                # the grid Optuna may choose from
SLIDE_FRACS = [0.5, 1.0]                    # slide = window * frac  (guarantees slide <= window)


def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path="../../train_val_test_mmsis_FINAL.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    mmsis = set(split_df.loc[split_df["split"] == which,"mmsi"])
    return mmsis
 
# All vessels in each quarter (no MMSI split -- the split is by TIME).
# validation mmsis from the whole of 2024 so we dont validate on the mmsis saved for testing only
GLOB_val_mmsis = get_global_val_test_mmsis(which="validation")
GLOB_test_mmsis = get_global_val_test_mmsis(which="test")
print(f"{len(GLOB_val_mmsis)} mmsis are reserved for validation, and {len(GLOB_test_mmsis)} are reserved for testing. We do not tune on these!")
all_mmsis_in_tuning = all_mmsis_in(TUNING_FILES)

tuning_mmsis = all_mmsis_in_tuning - GLOB_val_mmsis - GLOB_test_mmsis # REMOVE all validation and test mmsis, so these vessel are not seen by the tuning.
assert tuning_mmsis.isdisjoint(GLOB_val_mmsis), "Tuning MMSIS include val MMSIs!"
assert tuning_mmsis.isdisjoint(GLOB_test_mmsis), "Tuning MMSIS include test MMSIs!"

print(f"MMSIs available for tuning after excluding val/test mmsis training file: {len(tuning_mmsis)}")

# Split tuning mmsis in 80/20 for tuning
tuning_mmsis = np.array(list(tuning_mmsis))
rng = np.random.default_rng(42)
rng.shuffle(tuning_mmsis)

# Split into train test and validation set by mmsi so that no vessel appear in both.
n = len(tuning_mmsis)
train_mmsis = set(tuning_mmsis[:int(0.80*n)])
val_mmsis   = set(tuning_mmsis[int(0.80*n):])

le = LabelEncoder().fit(GEARS)
num_classes = len(le.classes_)

# ---------- windowing (parametrized + cached) ----------
def add_features_and_remove_val_test_mmsis(df):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df[df["report"].isin(GEARS)]          # <-- use the column that holds gear strings
    df = df[df["mmsi"].isin(tuning_mmsis)]
    m = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    return df

def iter_windows(arr, window, slide):
    n = len(arr)
    if n < window:
        pad = np.zeros((window - n, arr.shape[1]), dtype=np.float32)
        yield np.vstack([arr, pad]); return
    starts = list(range(0, n - window + 1, slide))
    if starts[-1] != n - window:
        starts.append(n - window)
    for s in starts:
        yield arr[s:s + window]

def build_windows(files, window, slide):
    X, y, groups = [], [], []
    for f in files:
        df = add_features_and_remove_val_test_mmsis(pd.read_parquet(f, engine="pyarrow"))
        for _, d in df.groupby("segment_id", sort=False):
            d = d.sort_values("date_time_utc")
            gear, mmsi = d["report"].iloc[0], d["mmsi"].iloc[0]
            arr = d[FEATURES].to_numpy(dtype=np.float32)
            for w in iter_windows(arr, window, slide):
                X.append(w); y.append(gear); groups.append(mmsi)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)

def cache_path(window, slide):
    return CACHE_DIR / f"win_w{window}_s{slide}.npz"

def build_all_caches():
    for w in WINDOWS:
        for fr in SLIDE_FRACS:
            s = max(1, int(w * fr))
            p = cache_path(w, s)
            if p.exists():
                continue
            X, y, g = build_windows(TUNING_FILES, w, s)
            np.savez(p, X=X, y=y, groups=g)
            print(f"cached w{w} s{s}: {X.shape}, {pd.Series(y).value_counts().to_dict()}")

build_all_caches()

# ---------- consistent grouped split (test stays sealed) ----------

# ---------- model (depth tunable, pooling guarded) ----------
def build_model(cfg, input_shape, num_classes):
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=input_shape)])
    cur_len, filters = input_shape[0], cfg["filters0"]
    for _ in range(cfg["n_conv"]):
        m.add(tf.keras.layers.Conv1D(filters, cfg["kernel"], padding="same", activation="relu"))
        m.add(tf.keras.layers.BatchNormalization())
        if cur_len // 2 >= 8:
            m.add(tf.keras.layers.MaxPooling1D(2)); cur_len //= 2
        filters = min(filters * 2, cfg["max_filters"])
    m.add(tf.keras.layers.GlobalAveragePooling1D())
    m.add(tf.keras.layers.Dense(cfg["dense"], activation="relu"))
    m.add(tf.keras.layers.Dropout(cfg["dropout"]))
    m.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return m

# ---------- objective ----------
def objective(trial):
    window     = trial.suggest_categorical("window", WINDOWS)
    slide_frac = trial.suggest_categorical("slide_frac", SLIDE_FRACS)
    slide      = max(1, int(window * slide_frac))

    data = np.load(cache_path(window, slide), allow_pickle=True)
    X, y, groups = data["X"], data["y"], data["groups"]
    y_enc = le.transform(y)

    train_mask = np.isin(groups, np.fromiter(train_mmsis, dtype=np.int64))
    val_mask   = np.isin(groups, np.fromiter(val_mmsis,   dtype=np.int64))
    Xtr, ytr = X[train_mask], y_enc[train_mask]
    Xva, yva = X[val_mask],   y_enc[val_mask]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xva = scaler.transform(Xva.reshape(-1, Xva.shape[-1])).reshape(Xva.shape)

    uc = np.unique(ytr)
    class_weight = dict(zip(uc, compute_class_weight("balanced", classes=uc, y=ytr)))

    cfg = {
        "n_conv":      trial.suggest_int("n_conv", 2, 4),
        "filters0":    trial.suggest_categorical("filters0", [32, 64, 128]),
        "max_filters": trial.suggest_categorical("max_filters", [128, 256]),
        "kernel":      trial.suggest_categorical("kernel", [3, 5, 7]),
        "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
        "dense":       trial.suggest_categorical("dense", [32, 64, 128]),
        "lr":          trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch":       trial.suggest_categorical("batch", [32, 64, 128]),
    }
    tf.keras.backend.clear_session()
    model = build_model(cfg, (window, len(FEATURES)), num_classes)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30,
              batch_size=cfg["batch"], class_weight=class_weight, callbacks=[es], verbose=0)

    pred = model.predict(Xva, verbose=0).argmax(1)
    return f1_score(yva, pred, average="macro", zero_division=0)

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            storage="sqlite:///optuna_gear_cnn.db",
                            study_name="gear_cnn", load_if_exists=True)
study.optimize(objective, n_trials=40)

print("Best macro-F1:", study.best_value)
print("Best params:", study.best_params)
json.dump({"best_value": study.best_value, "best_params": study.best_params},
          open("best_params_gear_cnn.json", "w"), indent=2)