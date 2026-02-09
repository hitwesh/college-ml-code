import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = 227
BATCH_SIZE = 32
EPOCHS = 5

base_dir = "Microplastic_Split"

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    valid_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes

model = Sequential()

model.add(Conv2D(96, (11,11), strides=4, activation="relu",
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(MaxPooling2D((3,3), strides=2))

model.add(Conv2D(256, (5,5), padding="same", activation="relu"))
model.add(MaxPooling2D((3,3), strides=2))

model.add(Conv2D(384, (3,3), padding="same", activation="relu"))
model.add(Conv2D(384, (3,3), padding="same", activation="relu"))
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D((3,3), strides=2))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer=Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_gen, epochs=EPOCHS, validation_data=valid_gen)

loss, acc = model.evaluate(test_gen)
print("Accuracy:", acc)

y_pred = model.predict(test_gen)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
