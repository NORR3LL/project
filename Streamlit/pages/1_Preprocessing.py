import streamlit as st
import pandas as pd
import os
import shutil
import time
from scripts.Reformator import convert_and_resize_images, augment_images, split_data

# Set page configuration
st.set_page_config(
    page_title="Upload Your Dataset",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add CSS styling
st.markdown(
    """
    <style>
    .header-style {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .sub-header-style {
        font-size: 18px;
        color: #555;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header-style">Preprocessing</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-style">Upload your dataset and prepare it for training with ease.</div>', unsafe_allow_html=True)

# File upload form
with st.form(key='preprocessing_form'):
    st.markdown("### 📤 Upload Dataset")
    st.write("Upload a zip file containing images for preprocessing.")
    
    uploaded_file = st.file_uploader("Choose a zip file", type="zip", key=1)
    target_size = st.number_input('📏 Target Image Size', value=256, help="Specify the desired size to resize the images.")
    
    st.markdown("### 🧩 Dataset Split Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_size = st.number_input('Train Data Size', value=0.7, min_value=0.0, max_value=1.0, step=0.01)
    with col2:
        val_size = st.number_input('Validation Data Size', value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    with col3:
        test_size = st.number_input('Test Data Size', value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    
    submit = st.form_submit_button('🚀 Start Preprocessing')

if submit:
    start_time = time.time()
    try:
        st.info("Processing started... Please wait.")

        if os.path.exists('outputs/Bronze/content/lung_ct_augmented'):
            st.write('🗂️ Files already exist. Starting augmentation...')
            if os.path.exists('outputs/Silver/train') and os.path.exists('outputs/Silver/val') and os.path.exists('outputs/Silver/test'):
                st.write('✨ Images Dataset Done. Starting data augmenting...')
                if os.path.exists('outputs/Gold/train') and os.path.exists('outputs/Gold/val') and os.path.exists('outputs/Gold/test'):
                    st.success('✅ Images already augmented.')
                else:
                    augment_images()
                    st.success('✅ Images successfully augmented.')
            else:
                split_data(train_split=train_size, val_split=val_size, test_split=test_size)
                st.write('✨ Split Done. Starting dataset Augmenting...')
                augment_images()
                st.success('✅ Images successfully split.')
        else:
            # Process the zip file
            st.write('🔄 Processing images...')
            convert_and_resize_images(uploaded_file, target_size=target_size)
            st.write('✨ Images augmented. Starting dataset splitting...')
            split_data(train_split=train_size, val_split=val_size, test_split=test_size)
            st.write('✨ Images resized. Starting augmentation...')
            augment_images()


            st.success('✅ Images successfully split.')

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    end_time = time.time()
    # Convert total time to hours, minutes, and seconds
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    st.write(f'⏱️ Processing completed in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')

# List the structure of the Gold folder
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        st.write(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            st.write(f'{subindent}{f}')

# Delete the uploaded files
def delete_files(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                st.success(f"🗑️ Deleted folder: {item_path}")
            except Exception as e:
                st.error(f"Error deleting folder {item_path}: {e}")

if st.button('🗑️ Delete Preprocessed Files'):
    delete_files('outputs/Bronze')
    delete_files('outputs/Silver')
    delete_files('outputs/Gold')
    st.success('✅ All preprocessed files deleted successfully.')
