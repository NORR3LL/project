import streamlit as st
import pandas as pd
import numpy as np
from scripts.Reformator import convert_and_resize_images
import time
import os

st.session_state.session_id = 0

# Load data
st.title('My first app')

with st.form(key='my_form',):
    st.write('Upload a zip file containing images')
    # Loading zip file
    uploaded_file = st.file_uploader("Choose a file", type="zip")
    target_size = st.number_input('Target size', value=int(256))
    submit=st.form_submit_button('Submit Zipfile')
 

if submit:
    if uploaded_file:
        uploaded_file.key = st.session_state.session_id
        try:
            
            # Process the zip file
            convert_and_resize_images(uploaded_file,target_size=target_size)
            st.write('Images processed successfully')
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a zip file before submitting.")
 

# delete the uploaded file
def delete_files():
    for file in os.listdir("outputs\\Bronze"):
        #ignore txt file
        os.remove(os.path.join("outputs\\Bronze", file))
    os.remove('outputs\\Bronze\\content') 


if st.button('Delete files'):
    delete_files()
    st.write('Files deleted successfully')