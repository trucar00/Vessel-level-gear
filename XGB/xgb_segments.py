import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt



# Download latest version
#path = kagglehub.dataset_download("sjleshrac/airlines-customer-satisfaction")
#print(path)

steam = pd.read_csv(f"Data/all_steaming_segs.csv")
longline = pd.read_csv("Data/line_fishing_segs.csv")
trawl = pd.read_csv("Data/trawl_fishing_segs.csv")

steam["label"] = "steam"
steam = steam.drop(columns=["steaming", "Unnamed: 0"])

df = pd.concat([steam, trawl, longline], ignore_index=True)
print("Class distribution:\n", df["label"].value_counts())
print("\nDtypes:\n", df.dtypes)

drop_cols = ["mmsi", "gear", "trajectory_id", "segment_id", "mean_dt", "std_dt"]

df = df.drop(columns=drop_cols)

X = df.drop(["label"], axis=1)

le = LabelEncoder()
y = le.fit_transform(df["label"])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

cv_params = {
    "max_depth":        [4, 6, 8],
    "min_child_weight": [1, 3, 5],
    "learning_rate":    [0.05, 0.1, 0.2],
    "n_estimators":     [200, 400],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
}

scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# -----------------------------------------------------------------------------
# 4. Grid-searched XGB baseline
# -----------------------------------------------------------------------------
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    tree_method="hist",          # fast, good default
    eval_metric="mlogloss",
    random_state=0,
    n_jobs=-1,
)

xgb_cv = GridSearchCV(
    xgb,
    cv_params,
    scoring=scoring,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    refit="f1_macro",
    n_jobs=-1,
    verbose=1,
)

sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

xgb_cv.fit(X_train, y_train, sample_weight=sample_weight_train)

print("\nBest params:", xgb_cv.best_params_)
print(f"Best CV f1_macro: {xgb_cv.best_score_:.4f}")

# -----------------------------------------------------------------------------
# 5. Evaluate on held-out test set
# -----------------------------------------------------------------------------
y_pred = xgb_cv.predict(X_test)

print(f"\nAccuracy:        {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall    (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1        (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1    (weighted):  {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# -----------------------------------------------------------------------------
# 6. Confusion matrix + feature importance
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, normalize="true")
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    cmap="Blues", values_format=".2f"
)
plt.title("Normalized confusion matrix (row = true class)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plot_importance(xgb_cv.best_estimator_, importance_type="gain",
                max_num_features=20, ax=ax)
plt.tight_layout()
plt.show()