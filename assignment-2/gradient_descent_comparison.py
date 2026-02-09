#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

data = load_breast_cancer()

X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

def create_model(input_dim):
    model = Sequential([
        Dense(8, activation='relu', input_shape=(input_dim,)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


print("\nBatch Gradient Descent")

model_batch = create_model(X.shape[1])

model_batch.fit(
    X, y,
    epochs=100,
    batch_size=len(X),
    verbose=0
)

loss_b, acc_b = model_batch.evaluate(X, y, verbose=0)

print("Accuracy:", acc_b)


print("\nStochastic Gradient Descent")

model_sgd = create_model(X.shape[1])

model_sgd.fit(
    X, y,
    epochs=100,
    batch_size=1,
    verbose=0
)

loss_s, acc_s = model_sgd.evaluate(X, y, verbose=0)

print("Accuracy:", acc_s)


print("\nMini-Batch Gradient Descent")

model_mbgd = create_model(X.shape[1])

model_mbgd.fit(
    X, y,
    epochs=100,
    batch_size=32,
    verbose=0
)

loss_m, acc_m = model_mbgd.evaluate(X, y, verbose=0)

print("Accuracy:", acc_m)


print("\nFinal Comparison")
print("Batch GD Accuracy:", acc_b)
print("SGD Accuracy:", acc_s)
print("Mini-Batch GD Accuracy:", acc_m)
