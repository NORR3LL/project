import streamlit as st
import os
from PIL import Image
from torchvision import transforms
import zipfile as zip
import random
import shutil


def convert_and_resize_images(zipfile, target_size):
    # create temp folder to store the zip file
    
    # create a folder to store the extracted images
    with zip.ZipFile(zipfile, 'r') as zip_ref:
        zip_ref.extractall('outputs/Bronze')
    zipfile.close()


def augment_images(target_size=(256, 256)):

    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    for subdir, _, files in os.walk('outputs\\Silver'):
        #ignore the test folder
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                #replace Bronze in subdir to Silver
                file_path = os.path.join(subdir, file_name)
                save_subdir= subdir.replace('Silver', 'Gold')
                save_path= os.path.join(save_subdir, file_name)
                try:
                    if not os.path.exists(save_subdir):
                        os.makedirs(save_subdir)
                        
                    if 'test' in subdir:
                        with Image.open(file_path) as img:
                            if img.mode != 'L':
                                img = img.convert('L')
                            img = img.resize(target_size, Image.LANCZOS)  # resize
                            img.save(save_path)  # save processed image
                            print(f'Processed {file_path}')

                    else:
                        with Image.open(file_path) as img:
                            if img.mode != 'L':  # if it is not grey
                                img = img.convert('L')  # grey
                            img = img.resize(target_size, Image.LANCZOS)  # resize
                            img = augmentation_transforms(img)
                            img.save(save_path)  # save processed image
                            print(f'Processed {file_path}')

                except Exception as e:
                    st.error(f"Error processing {file_path}: {e}")
 


# Function to split and copy images
def split_data(train_split, val_split, test_split):
    source = r'outputs\Bronze\content\lung_ct_augmented'
    target_dirs = {
        "train": r'outputs\Silver\train',
        "val": r'outputs\Silver\val',
        "test": r'outputs\Silver\test'
    }

    # Parameters for splitting
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15

    # Check that splits add up to 1
    assert train_split + val_split + test_split == 1, "Splits do not add up to 1!"

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
 

        file_splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }


        # Iterate over splits
        for split_name in ["train", "val", "test"]:
            split_files = file_splits[split_name]
            target_dir = os.path.join(target_dirs[split_name], category)

            # Create directory if it doesn't exist
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            # Copy files to the appropriate directory
            for file in split_files:
                src = os.path.join(category_path, file)
                dst = os.path.join(target_dir, file)
                shutil.copy2(src, dst)
                print(f'Copied {src} to {dst}')