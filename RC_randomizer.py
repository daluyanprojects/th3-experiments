import os
import random
import shutil
import math

def split_rs_dataset(source_dir, output_dir, train_ratio=0.7, seed=42):
    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path)
                 if os.path.isfile(os.path.join(class_path, f))]

        random.shuffle(files)

        split_index = math.floor(len(files) * train_ratio)

        train_files = files[:split_index]
        val_files = files[split_index:]

        for f in train_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(train_dir, f)
            )

        for f in val_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(val_dir, f)
            )

        print(f"{class_name}: {len(train_files)} train | {len(val_files)} validation")

    print("\n70/30 split completed!")


def split_rs_dataset_three_way(source_dir, output_dir,
                               train_ratio=0.5, val_ratio=0.25, seed=42):
    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path)
                 if os.path.isfile(os.path.join(class_path, f))]

        random.shuffle(files)

        num_files = len(files)
        train_end = math.floor(num_files * train_ratio)
        val_end = math.floor(num_files * (train_ratio + val_ratio))

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for f in train_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(train_dir, f)
            )

        for f in val_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(val_dir, f)
            )

        for f in test_files:
            shutil.copy(
                os.path.join(class_path, f),
                os.path.join(test_dir, f)
            )

        print(f"{class_name}: {len(train_files)} train | "
              f"{len(val_files)} validation | {len(test_files)} test")

    print("\n50/25/25 pooled split completed!")
