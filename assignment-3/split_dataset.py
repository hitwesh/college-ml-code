import os
import shutil
import random

source_dir = r"C:\dev\ml-project\College-ml-code\assignment-3\Microplastics and algae"
base_dir = "Microplastic_Split"

train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(source_dir)

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    valid_end = int(len(images) * (train_ratio + valid_ratio))

    train_imgs = images[:train_end]
    valid_imgs = images[train_end:valid_end]
    test_imgs = images[valid_end:]

    for folder in ["train", "valid", "test"]:
        os.makedirs(os.path.join(base_dir, folder, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "train", cls, img)
        )

    for img in valid_imgs:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "valid", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "test", cls, img)
        )

print("Dataset split completed.")
