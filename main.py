import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from pathlib import Path
from config.const import DATASET, MODELS_DIR

# Load dataset
df = pd.read_csv(DATASET, header=None, encoding="latin-1")
df = df.iloc[1:, :]  
df.columns = ["target", "text"]

# Keep positive and negative tweets
df = df[df["target"] != "2"]       
df["target"] = df["target"].map({"0": 0, "4": 1, "1": 1})  
df = df.reset_index()

# Test 1
# print(df.head())
# print(df["target"].value_counts()) 

# Preprocess
def _lowercase(text: str) -> str:
    return text.lower()

df["text"] = df["text"].apply(_lowercase)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"],
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

# Test 2
# print(len(X_train))
# print(len(df))
# print(len(X_test))
# print(len(y_test))
# print(len(y_test))


vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Test 3
# print("Train matrix shape:", X_train_vec.shape)
# print("Test matrix shape:", X_test_vec.shape)
# print("Sample feature names:", vec.get_feature_names_out()[:10])

# Training Models:
# Bernoulli
bnb = BernoulliNB()
bnb.fit(X_train_vec, y_train)
y_bnb_pred = bnb.predict(X_test_vec)

# SVC 
lsvc = LinearSVC(max_iter=1000, random_state=42)
lsvc.fit(X_train_vec, y_train)
y_lsvc_pred = lsvc.predict(X_test_vec)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
y_lr_pred = lr.predict(X_test_vec)

# joblib.dump(bnb, MODELS_DIR / "bnb.pkl")
# joblib.dump(lsvc, MODELS_DIR / "lsvc.pkl")
# joblib.dump(lr, MODELS_DIR / "lr.pkl")
# joblib.dump(vec, MODELS_DIR / "vectorizer.pkl")

# Test Accuracy
print(f"BNB Accuracy (test): {accuracy_score(y_test, y_bnb_pred) * 100:.2f}%")
print(f"SVC Accuracy (test): {accuracy_score(y_test, y_lsvc_pred) * 100:.2f}%")
print(f"LR Accuracy (test): {accuracy_score(y_test, y_lr_pred) * 100:.2f}%")

# Inference on sample texts (and show confidence %)
sample_texts = [
    "I love sisg!",
    "Hey that looks good! Actually it looks like trash!"
    "You suck! :)",
    "Damn that was pretty nice.",
    "He was really bad",
    "Life kinda hard rn",
    "Huawei sucks compared to Samsung",
    "I am going to kill someone",
  
]
sample_vec = vec.transform(sample_texts)
models = ["bnb", "lsvc", "lr"]

load_models = {}
for name in models:
    model_path = MODELS_DIR / f"{name}.pkl"
    try:
        load_models[name] = joblib.load(model_path)
    except Exception:
        print(f"Could not load {model_path}. Using in-memory trained model for '{name}'.")
        fallback = {"bnb": bnb, "lsvc": lsvc, "lr": lr}
        load_models[name] = fallback[name]


def _get_preds(model, X):
    preds = model.predict(X)
    return preds.astype(int)


def _get_confidences(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        else:
            return probs.max(axis=1)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 1:
            return 1.0 / (1.0 + np.exp(-scores))
        else:
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp = np.exp(shifted)
            softm = exp / np.sum(exp, axis=1, keepdims=True)
            if softm.shape[1] == 2:
                return softm[:, 1]
            return softm.max(axis=1)

    return np.full(X.shape[0], np.nan)


predictions = {name: _get_preds(mdl, sample_vec) for name, mdl in load_models.items()}
confidences = {name: _get_confidences(mdl, sample_vec) for name, mdl in load_models.items()}

for i, text in enumerate(sample_texts):
    print(f"\nText: {text}")
    for name in models:
        pred = int(predictions[name][i])
        sentiment = "Positive" if pred == 1 else "Negative"
        conf = confidences[name][i]
        if conf is not None and not (np.isnan(conf)):
            conf_pct = f"{conf * 100:.2f}%"
        else:
            conf_pct = "N/A"
        print(f"{name}: {sentiment} ({conf_pct})")
