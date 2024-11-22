import streamlit as st
import os
from PIL import Image

# set size
target_size = (256, 256)

# def processing progress
def convert_and_resize_images(folder, target_size):
    """
    Convert all images in the folder to grayscale and resize them
    """
    processed_files = []
    for subdir, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(subdir, file_name)

                try:
                    with Image.open(file_path) as img:
                        if img.mode != 'L':  # if it is not grey
                            img = img.convert('L')  # grey
                        img = img.resize(target_size, Image.LANCZOS)  # resize
                        img.save(file_path)  # save processed image
                        processed_files.append(file_path)
                except Exception as e:
                    st.error(f"Error processing {file_path}: {e}")
    return processed_files

# Streamlit
st.title("Batch Preprocessing of Lung CT Images")
st.write("This tool will batch process lung CT images in a folder, supporting conversion to grayscale and resizing of the images.")

# folder path
folder_path = st.text_input("Enter the folder path：")

if st.button("starting processing"):
    if folder_path and os.path.exists(folder_path):
        st.write(f"processing folder：{folder_path}...")
        processed_files = convert_and_resize_images(folder_path, target_size)
        st.success(f"Done！已处理 {len(processed_files)} image")
        st.write("Processed image：")
        st.write(processed_files)
    else:
        st.error("Please enter a valid folder path!")

