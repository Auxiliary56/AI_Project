import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1) LOAD DATA
# =========================

CSV_PATH =r"C:\Users\DELL\Downloads\archive (1)\UNSW_NB15_training-set.csv"  
df = pd.read_csv(CSV_PATH)

 

if "label" not in df.columns:
    raise ValueError("Couldn't find 'label' column. Please check your CSV columns.")

y = df["label"].astype(int)

# Drop columns that can cause leakage / are not inputs
# attack_cat is multi-class label; keep ONLY for multi-class tasks, drop for binary benign vs malicious

drop_cols = [c for c in ["label", "attack_cat"] if c in df.columns]
X = df.drop(columns=drop_cols)


# =========================
# 2) BASIC COLUMN SETUP
# =========================

candidate_cat = [c for c in ["proto", "service", "state"] if c in X.columns]

# Everything else numeric

num_cols = [c for c in X.columns if c not in candidate_cat]

# Train-test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# 3) PREPROCESSING PIPELINE
# =========================
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, candidate_cat)
    ],
    remainder="drop"
)


# =========================
# 4) GRAPH 1 — CORRELATION "HEATMAP" (NUMERIC ONLY)
# =========================
# Correlation is defined for numeric features; we’ll show a compact heatmap for top correlated numeric features
tmp = df[num_cols + ["label"]].copy()
tmp = tmp.fillna(tmp.median(numeric_only=True))
corr = tmp.corr(numeric_only=True)

# Get top numeric features correlated with label (absolute correlation)
label_corr = corr["label"].drop("label").abs().sort_values(ascending=False)
top_for_heatmap = label_corr.head(15).index.tolist()  # show 15 for readability

heat = corr.loc[top_for_heatmap + ["label"], top_for_heatmap + ["label"]]

plt.figure()
plt.imshow(heat.values, aspect="auto")
plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Correlation Heatmap (Top 15 numeric features + label)")
plt.colorbar()
plt.tight_layout()
plt.show()


# =========================
# 5) RANDOM FOREST MODEL (ALL FEATURES) + GRAPH 2 IMPORTANCE
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model_all = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", rf)
])

model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)

print("\n=== Model with ALL features ===")
print("Accuracy:", accuracy_score(y_test, pred_all))
print(classification_report(y_test, pred_all, digits=4))


# ---- Extract feature names after preprocessing
prep_fitted = model_all.named_steps["prep"]
rf_fitted = model_all.named_steps["rf"]

# Numeric feature names stay same
num_feature_names = num_cols

# Categorical feature names expand into one-hot columns
cat_feature_names = []
if len(candidate_cat) > 0:
    ohe = prep_fitted.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(candidate_cat).tolist()

all_feature_names = num_feature_names + cat_feature_names

importances = rf_fitted.feature_importances_
imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

# Plot top 20 transformed features (includes one-hot)
topN = 20
plt.figure()
plt.bar(range(topN), imp_series.head(topN).values)
plt.xticks(range(topN), imp_series.head(topN).index, rotation=90)
plt.title("Random Forest Feature Importance (Top 20)")
plt.tight_layout()
plt.show()


# =========================
# 6) SELECT TOP 10 "BASE" FEATURES (GROUPING ONE-HOT BACK)
# =========================
# We want 10 FEATURES total (not 10 one-hot columns).
# So we group:
# - numeric: keep as-is
# - categorical: sum importances for all one-hot columns belonging to the same original column (proto/service/state)

base_scores = {}

# Numeric: direct
for f in num_cols:
    base_scores[f] = base_scores.get(f, 0) + imp_series.get(f, 0)

# Categorical: sum all one-hot parts back to their base feature
for c in candidate_cat:
    # one-hot columns start with "c_" in get_feature_names_out format
    matching = [name for name in imp_series.index if name.startswith(c + "_")]
    base_scores[c] = float(imp_series[matching].sum()) if matching else 0.0

base_rank = pd.Series(base_scores).sort_values(ascending=False)

top10_features = base_rank.head(10).index.tolist()
print("\nTop 10 selected base features:")
print(top10_features)


# =========================
# 7) GRAPH 3 — ACCURACY VS NUMBER OF FEATURES
# =========================

ks = list(range(5, min(31, len(base_rank) + 1)))
accs = []

for k in ks:
    selected = base_rank.head(k).index.tolist()

    sel_cat = [c for c in candidate_cat if c in selected]
    sel_num = [c for c in selected if c not in candidate_cat]

    preprocess_k = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, sel_num),
            ("cat", categorical_pipe, sel_cat)
        ],
        remainder="drop"
    )

    model_k = Pipeline(steps=[
        ("prep", preprocess_k),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    model_k.fit(X_train[selected], y_train)
    pred_k = model_k.predict(X_test[selected])
    accs.append(accuracy_score(y_test, pred_k))

plt.figure()
plt.plot(ks, accs, marker="o")
plt.xticks(ks, rotation=90)
plt.title("Accuracy vs Number of Selected Features (Random Forest)")
plt.xlabel("Number of features (k)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()


# =========================
# 8) FINAL MODEL WITH TOP 10 + CONFUSION MATRIX
# =========================
selected = top10_features
sel_cat = [c for c in candidate_cat if c in selected]
sel_num = [c for c in selected if c not in candidate_cat]

preprocess_10 = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, sel_num),
        ("cat", categorical_pipe, sel_cat)
    ],
    remainder="drop"
)

model_10 = Pipeline(steps=[
    ("prep", preprocess_10),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

model_10.fit(X_train[selected], y_train)
pred_10 = model_10.predict(X_test[selected])

print("\n=== FINAL MODEL with TOP 10 features ===")
print("Accuracy:", accuracy_score(y_test, pred_10))
print(classification_report(y_test, pred_10, digits=4))

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, pred_10)
plt.title("Confusion Matrix (Top 10 Features)")
plt.tight_layout()
plt.show()