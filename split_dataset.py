import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = 'poultry_diseases'  
BASE_OUTPUT_DIR = 'dataset_split'

TRAIN_DIR = os.path.join(BASE_OUTPUT_DIR, 'train')
TEST_DIR = os.path.join(BASE_OUTPUT_DIR, 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TEST_DIR, cls, img))

print("✅ Dataset successfully split into train and test folders.")
