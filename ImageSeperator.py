import os
import shutil
import random
from pathlib import Path

# Set the paths
root_dir = 'C:/uoft/1517/content/content/lung_ct_augmented'  # Replace with the path to your root directory
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')

# Create the target directories
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Parameters for splitting
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Check that splits add up to 1
assert train_split + val_split + test_split == 1, "Splits do not add up to 1!"

# Function to split and copy images
def split_data(source, train_dir, val_dir, test_dir):
    for category in os.listdir(source):
        category_path = os.path.join(source, category)
        
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
        
        # Get list of files
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(files)

        # Calculate split indices
        train_end = int(len(files) * train_split)
        val_end = train_end + int(len(files) * val_split)

        # Split files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Create category directories in train, val, test
        for folder, file_set in zip([train_dir, val_dir, test_dir], [train_files, val_files, test_files]):
            target_category_dir = os.path.join(folder, category)
            os.makedirs(target_category_dir, exist_ok=True)

            # Copy files
            for file in file_set:
                src = os.path.join(category_path, file)
                dst = os.path.join(target_category_dir, file)
                shutil.copy2(src, dst)

# Run the split
split_data(root_dir, train_dir, val_dir, test_dir)



