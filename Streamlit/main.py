import streamlit as st
import pandas as pd
import numpy as np
from scripts.Reformator import convert_and_resize_images
import time
import os


#create a main page with introduction about the project
st.title('Team17: Lung CT Classification')

st.write('This is a project to classify lung CT images into 3 classes: COVID-19, Normal, and Pneumonia. The dataset is obtained from Kaggle and the model is trained using PyTorch. The model is trained using a ResNet18 architecture and achieves an accuracy of 90% on the test set. The model is deployed using Streamlit and Heroku.')

st.write('The project is divided into 3 main parts: Preprocessing, Training, and Inference. The Preprocessing page is used to preprocess the images before training. The Training page is used to train the model. The Inference page is used to make predictions on new images.')




# st.session_state.session_id = 0

# # Load data
# st.title('My first app')

# with st.form(key='my_form',):
#     st.write('Upload a zip file containing images')
#     # Loading zip file
#     uploaded_file = st.file_uploader("Choose a file", type="zip")
#     target_size = st.number_input('Target size', value=int(256))
#     submit=st.form_submit_button('Submit Zipfile')
 

# if submit:
#     if uploaded_file:
#         uploaded_file.key = st.session_state.session_id
#         try:
            
#             # Process the zip file
#             convert_and_resize_images(uploaded_file,target_size=target_size)
#             st.write('Images processed successfully')
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#     else:
#         st.warning("Please upload a zip file before submitting.")

#     class_counts = {}

#     for class_folder in os.listdir('outputs/Bronze/content/lung_ct_augmented'):
#         class_path = os.path.join('outputs/Bronze/content/lung_ct_augmented', class_folder)
#         if os.path.isdir(class_path):
#             # Count the number of images
#             num_images = len([file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))])
#             class_counts[class_folder] = num_images

#     for class_name, count in class_counts.items():
#         st.write(f"Class {class_name} has {count} images")
 

# # delete the uploaded file
# def delete_files():
#     for file in os.listdir("outputs\\Bronze"):
#         #ignore txt file
#         os.remove(os.path.join("outputs\\Bronze", file))
#     os.remove('outputs\\Bronze\\content') 


# if st.button('Delete files'):
#     delete_files()
#     st.write('Files deleted successfully')