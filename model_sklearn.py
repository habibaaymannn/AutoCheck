"""
Simple scikit-learn model — Iris flower classification
with SKLearnCheckpointManager save / load / resume demo
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from checkpointManager.SKLearnCheckpointManager import SKLearnCheckpointManager

ckpt_manager = SKLearnCheckpointManager()
SAVE_DIR = "checkpoints"

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
)

# ── Train ─────────────────────────────────────────────────────────────────────
model.fit(X_train, y_train)

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Save checkpoint ───────────────────────────────────────────────────────────
ckpt_path = ckpt_manager.save_checkpoint({
    "model":      model,
    "scaler":     scaler,          # not serialized to joblib — saved via metadata? no.
    "step":       1,
    "cv_mean":    float(cv_scores.mean()),
    "cv_std":     float(cv_scores.std()),
    "n_estimators": model.n_estimators,
    "max_depth":    model.max_depth,
}, SAVE_DIR)
print(f"\nCheckpoint saved → {ckpt_path}")

# ── Load checkpoint & resume ──────────────────────────────────────────────────
state = ckpt_manager.load_checkpoint(SAVE_DIR)
model = state["model"]          # fully restored RandomForestClassifier
print(f"Checkpoint loaded  (v{state['checkpoint_version']},"
      f" saved at: {state['session_info']['last_save_time']})")
print(f"  CV mean from checkpoint : {state['cv_mean']:.4f}")

print(state)

# ── Evaluate (using restored model) ───────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nTest accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Feature importance ────────────────────────────────────────────────────────
print("\nFeature importances:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name:<26} {importance:.4f}")

# ── Predict (sample) ──────────────────────────────────────────────────────────
sample = X_test[:5]
preds  = model.predict(sample)

print("\nSample predictions:")
for i, (pred, true) in enumerate(zip(preds, y_test[:5])):
    print(f"  [{i}] Predicted: {iris.target_names[pred]:<12}  Actual: {iris.target_names[true]}")