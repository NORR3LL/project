import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import shutil
from scripts.Reformator import convert_and_resize_images, augment_images, split_data  


st.title('Preprocessing')
st.session_state.session_id = 0

# Load data
st.title('My first app')

with st.form(key='my_form'):
    st.write('Upload a zip file containing images')
    # Loading zip file
    uploaded_file = st.file_uploader("Choose a file", type="zip", key=1)
    target_size = st.number_input('Image Scale', value=int(256))

    col1, col2,col3 = st.columns(3)
    with col1:
        train_size = st.number_input('Train Data size', value=0.7)
    with col2:
        val_size = st.number_input('Validation Datasize', value=0.15)
    with col3:
        test_size = st.number_input('Test Data size', value=0.15)

    submit=st.form_submit_button('Submit Zipfile')
 

if submit:
        start_time = time.time()
        try:
            if os.path.exists('outputs/Bronze/content/lung_ct_augmented'):
                st.write('Files already exist. Starting augmentation')
                if os.path.exists('outputs/Silver/content/lung_ct_augmented'):
                    st.write('Images augmented already finished.Starting split the Dataset to train,val and test')
                    if os.path.exists('outputs/Gold/train') and os.path.exists('outputs/Gold/val') and os.path.exists('outputs/Gold/test'):
                        st.write('Images already split')
                    else:
                        split_data(train_split=train_size, val_split=val_size, test_split=test_size)
                        st.write('Images split successfully')
                else:
                    augment_images()
                    st.write('Images augmented successfully.Starting split the Dataset to train,val and test')
                    split_data(train_split=train_size, val_split=val_size, test_split=test_size)
                    st.write('Images split successfully')
 
            else:
            # Process the zip file
                convert_and_resize_images(uploaded_file,target_size=target_size)
                st.write('Images processed successfully. Starting augmentation')
                augment_images()
                st.write('Images augmented successfully.Starting split the Dataset to train,val and test')
                split_data(train_split=train_size, val_split=val_size, test_split=test_size)
                st.write('Images split successfully')

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        end_time = time.time()
        #coverting time to hours, minutes and seconds
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        st.write(f'Images processed in {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds')
 
 # list the gold folder sturtcure
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        st.write('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            st.write('{}{}'.format(subindent, f))

# delete the uploaded file
def delete_files(path):
    for item in os.listdir(path):
            item_path = os.path.join(path, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                try:
                    # Recursively delete the directory
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                except Exception as e:
                    print(f"Error deleting folder {item_path}: {e}")

if st.button('Delete files'):
    delete_files('outputs/Bronze')
    delete_files('outputs/Silver')
    delete_files('outputs/Gold')
    st.write('Files deleted successfully')