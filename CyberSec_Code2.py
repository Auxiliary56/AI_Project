import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1) SET PATH + TOP 10 FEATURES
# =========================
CSV_PATH = r"C:\Users\DELL\Downloads\archive (1)\UNSW_NB15_training-set.csv"

# Paste YOUR top 10 feature names here exactly as they appear in your CSV
TOP10_FEATURES = [
    "dur", "proto", "service", "spkts", "dpkts",
    "sbytes", "dbytes", "smean", "dmean", "ct_srv_src"
]

LABEL_COL = "label"  # 0 = benign/normal, 1 = attack (binary)


# =========================
# 2) OUTLIER CLIPPING FUNCTION (IQR CAPPING)
# =========================
def iqr_clip_train_test(X_train_df, X_test_df, numeric_cols, k=1.5):
    """
    Clips numeric outliers using IQR bounds computed on TRAIN only
    to avoid leakage.

    k=1.5 is standard. Increase to 3.0 if you want less aggressive clipping.
    """
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()

    for col in numeric_cols:
        # Convert to numeric safely
        tr = pd.to_numeric(X_train_df[col], errors="coerce")

        q1 = tr.quantile(0.25)
        q3 = tr.quantile(0.75)
        iqr = q3 - q1

        # If IQR is 0 (constant column), skip clipping
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce").clip(lower, upper)
        X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce").clip(lower, upper)

    return X_train_df, X_test_df


# =========================
# 3) LOAD DATA + KEEP ONLY TOP10 + LABEL
# =========================
df = pd.read_csv(CSV_PATH)

# Validate columns
missing = [c for c in TOP10_FEATURES + [LABEL_COL] if c not in df.columns]
if missing:
    raise ValueError(f"These columns are missing in your CSV: {missing}")

# Drop all other features (this is what you asked)
df_small = df[TOP10_FEATURES + [LABEL_COL]].copy()

# Separate X and y
X = df_small[TOP10_FEATURES]
y = df_small[LABEL_COL].astype(int)


# =========================
# 4) TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# =========================
# 5) DETECT CATEGORICAL VS NUMERIC (within top10)
# =========================
# In UNSW-NB15 these are often categorical if present:
possible_cat = {"proto", "service", "state"}
cat_cols = [c for c in TOP10_FEATURES if c in possible_cat]

# Everything else treat as numeric
num_cols = [c for c in TOP10_FEATURES if c not in cat_cols]


# =========================
# 6) NOISE HANDLING: OUTLIER CLIPPING (NUMERIC ONLY)
# =========================
X_train, X_test = iqr_clip_train_test(X_train, X_test, num_cols, k=1.5)


# =========================
# 7) PREPROCESSING + MODEL PIPELINE
# =========================
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # Remove constant/near-constant numeric columns (noise / no info)
    ("variance", VarianceThreshold(threshold=0.0)),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ],
    remainder="drop"
)

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", model)
])


# =========================
# 8) TRAIN + EVALUATE
# =========================
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
print("\n=== Top 10 Features Only (Noise handled) ===")
print("Top 10 features used:", TOP10_FEATURES)
print("Categorical used:", cat_cols)
print("Numeric used:", num_cols)
print("\nAccuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, pred, digits=4))

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix (Top 10 Features Only)")
plt.tight_layout()
plt.show()