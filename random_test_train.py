import os
import random
import shutil

input_dir = r"C:\Users\kevin\OneDrive\Desktop\Compiladors\flowers"

output_dir = "flowers_split"

train_ratio = 0.8

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")


train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(valid_extensions)
    ]

    total_images = len(images)
    print(f"{class_name}: {total_images} imágenes")

    random.shuffle(images)

    split_index = int(total_images * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy2(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy2(src, dst)

print("\nTrain y test creado existosamente")