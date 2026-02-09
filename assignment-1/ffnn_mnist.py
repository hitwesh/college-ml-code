#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load dataset (MNIST example)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Define Feed Forward Neural Network (MANDATORY)

model = Sequential([
    # Input Layer + Hidden Layer 1
    Dense(128, activation='relu', input_shape=(784,)),
    # Hidden Layer 2
    Dense(64, activation='relu'),
    # Output Layer
    Dense(10, activation='softmax'),
])

# ENFORCEMENT CHECK
if model is None:
    raise RuntimeError(
        "Feed Forward Neural Network is NOT defined.\n"
        "Please define the model before compiling."
    )

# 4. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Model summary
model.summary()

# 6. Train the model
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 7. Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\nTest Accuracy:", test_accuracy)
