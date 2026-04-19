"""
Simple TensorFlow/Keras model — Iris flower classification
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy:.4f}")

# ── Predict (sample) ──────────────────────────────────────────────────────────
sample = X_test[:5]
preds = model.predict(sample)
predicted_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test[:5], axis=1)

print("\nSample predictions:")
for i, (pred, true) in enumerate(zip(predicted_classes, true_classes)):
    print(f"  [{i}] Predicted: {iris.target_names[pred]:<12}  Actual: {iris.target_names[true]}")
