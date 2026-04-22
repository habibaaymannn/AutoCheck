"""
mlTrainingKerasModel.py
Real Keras training — MNIST digit classifier.
Zero awareness of AutoCheck — just plain Keras / TensorFlow code.

Press Ctrl+C at any point to pause.
AutoCheck saves checkpoint. Run again to resume.

Requirements:
    pip install tensorflow
"""

import tensorflow as tf
from tensorflow import keras

# ── Model ─────────────────────────────────────────────────────────────

def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")
    return model


# ── Setup ─────────────────────────────────────────────────────────────

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
x_train = (x_train - 127.5) / 127.5          # normalise to [-1, 1]

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(60_000, reshuffle_each_iteration=True)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

model     = build_model()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
criterion = keras.losses.SparseCategoricalCrossentropy()

# Learning-rate schedule: halve LR every 3 epochs (mirrors StepLR(step=3, gamma=0.5))
def get_lr(current_epoch: int) -> float:
    return 1e-3 * (0.5 ** (current_epoch // 3))

NUM_EPOCHS  = 20

print("Starting MNIST training... (press Ctrl+C to pause)")
print(f"{'epoch':>6} {'batch_idx':>10} {'global_step':>12} {'loss':>10}")
print("-" * 45)

# ── Training loop ─────────────────────────────────────────────────────

epoch     = 0
batch_idx = 0
global_step = 0

start_epoch = epoch

for epoch in range(start_epoch, NUM_EPOCHS):

    # Apply LR schedule at the start of each epoch
    optimizer.learning_rate.assign(get_lr(epoch))

    resume_from = batch_idx if epoch == start_epoch else 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):

        if batch_idx < resume_from:
            continue

        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss        = criterion(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        global_step += 1

        if global_step % 100 == 0:
            print(
                f"{epoch:>6} {batch_idx:>10} {global_step:>12} "
                f"{float(loss):>10.4f}"
            )

    batch_idx = 0
    print(f"  -> epoch {epoch} done | lr={float(optimizer.learning_rate.numpy()):.6f}")

print("\nTraining complete!")
