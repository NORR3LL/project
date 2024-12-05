import streamlit as st
import pandas as pd
import numpy as np
from scripts.Reformator import convert_and_resize_images
import time
import os


import streamlit as st

#set the page name in the sidebar to 'Home'
st.set_page_config(page_title='Home', page_icon='🏠', layout='wide', initial_sidebar_state='expanded')

# Main Page with Introduction
st.title("Lung CT Classification Project 🫁")



st.markdown(
    """
    ### Welcome to the Lung CT Image Classification Platform!  
    This project focuses on classifying lung CT images into 5 distinct categories**:
    - **COVID-19**
    - **Normal**
    - **Bacterial Pneumonia**
    - **Lung Opacity**
    - **Viral Pneumonia**


    Using data from [Kaggle](https://www.kaggle.com/), we have developed a robust classification model powered by **PyTorch**.  
    Our pipeline incorporates the **ResNet50 architecture**, achieving an impressive **96% accuracy** on the test set.  
    The platform is deployed using **Streamlit** , making it accessible and user-friendly.
    """
)

st.divider()

st.markdown(
    """
    ### How the Project Works:
    The application is divided into **three main components** to streamline the process:
    1. **Preprocessing:**  
       - Prepare and clean the images before training.
    2. **Training:**  
       - Train the classification model using the preprocessed data.
    3. **Inference:**  
       - Make predictions on new CT images and classify them into the appropriate category.
    """
)

st.info("✨ Navigate through the pages in the sidebar to explore each part of the project in detail!")
