import tensorflow as tf
import numpy as np

from checkpointManager.KerasCheckpointManager import KerasCheckpointManager

# ── Create dummy data ───────────────────────────────
X = np.random.rand(100, 4)
y = np.sum(X, axis=1)

# ── Build model ────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse'
)

# ── Train a bit (IMPORTANT for optimizer state) ────
model.fit(X, y, epochs=2, verbose=0)

# ── Save checkpoint ────────────────────────────────
manager = KerasCheckpointManager("checkpoints", max_to_keep=3)

state = {
    "model": model,
    "epoch": 2,
    "loss": float(model.evaluate(X, y, verbose=0))
}

save_path = manager.save_checkpoint(state, "checkpoints")
print("Saved to:", save_path)

# ── Load checkpoint ────────────────────────────────
loaded = manager.load_checkpoint("checkpoints")

loaded_model = loaded["model"]

print("Loaded version:", loaded["checkpoint_version"])
print("Loaded metadata:", {k: v for k, v in loaded.items() if k not in ["model", "optimizer"]})

# ── Verify it actually works ───────────────────────
pred1 = model.predict(X[:5])
pred2 = loaded_model.predict(X[:5])

print("Predictions match:", np.allclose(pred1, pred2))