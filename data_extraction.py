import zipfile
import os
import shutil

zip_path = r"C:\Users\Jyoti\Downloads\archive (9).zip"
extract_to = "extracted_data"
final_dataset_path = "data"

# Step 1: Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Step 2: Define accurate source paths
train_src = os.path.join(extract_to, "New Plant Diseases Dataset(Augmented)","New Plant Diseases Dataset(Augmented)", "train")
valid_src = os.path.join(extract_to, "New Plant Diseases Dataset(Augmented)","New Plant Diseases Dataset(Augmented)", "valid")
test_src = os.path.join(extract_to, "test", "test")  

sources = [train_src, valid_src, test_src]
target_names = ["train", "valid", "test"]

# Step 3: Copy folders cleanly
for src, name in zip(sources, target_names):
    dst = os.path.join(final_dataset_path, name)
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"✅ Copied {name} to {dst}")
    else:
        print(f"❌ Folder '{name}' not found at: {src}")


