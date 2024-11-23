import streamlit as st
import os
from PIL import Image
from torchvision import transforms
import zipfile as zip


def convert_and_resize_images(zipfile, target_size):
    # create temp folder to store the zip file
    
    # Unzip the file 
    with zip.ZipFile(zipfile, 'r') as zip_ref:
        zip_ref.extractall('outputs/Bronze')
    
    zipfile.close()


    # processed_files = []
    # augmentation_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    # ])
    # for subdir, _, files in os.walk(folder):
    #     for file_name in files:
    #         if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
    #             file_path = os.path.join(subdir, file_name)
    #             print(file_path)

    #             # try:
    #             #     with Image.open(file_path) as img:
    #             #         if img.mode != 'L':  # if it is not grey
    #             #             img = img.convert('L')  # grey
    #             #         img = img.resize(target_size, Image.LANCZOS)  # resize
    #             #         img = augmentation_transforms(img)
    #             #         #save the image to a data folder
    #             #         img.save(file_path)  # save processed image
    #             #         processed_files.append(file_path)
    #             # except Exception as e:
    #             #     st.error(f"Error processing {file_path}: {e}")
 


