import os
import shutil

src_root = "dataset/images/raw-img"
dst_root = "dataset/images"

count = 0

for category in os.listdir(src_root):
    category_path = os.path.join(src_root, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            src_file = os.path.join(category_path, file)

            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                new_name = f"{category}_{count}.jpg"
                dst_file = os.path.join(dst_root, new_name)

                shutil.copy(src_file, dst_file)
                count += 1

print("Moved images:", count)