# user_program.py

import tensorflow as tf
import numpy as np

X = np.random.rand(100, 4)
y = np.sum(X, axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

for epoch in range(3):
    model.fit(X, y, epochs=1, verbose=0)

    # 👇 THIS is what your tracker must capture
    global_step = epoch