from scripts.trainners import Trainer,get_train_loaders,get_test_loader
from scripts.models import Model

import streamlit as st
import torch.nn as nn
import torch.optim as optim
import torch
import os


tab1, tab2 = st.tabs(["Model Training", "Model Evaluation"])

with tab1: 
    st.header("Model Training")
    st.write("This is the model training page")
    st.write("Please select the training and validation directories")

    with st.form(key="model_training"):
        train_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\train'
        val_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\val'
        batch_size = st.number_input("Batch Size", value=1)
        num_epochs = st.number_input("Number of Epochs", value=1)
        learning_rate = st.number_input("Learning Rate", value=0)
        patience = st.number_input("Patience", value=0)
        warmup_steps = st.number_input("Warmup Steps", value=0)
        submit=st.form_submit_button("Train Model")

        if submit:
            model = Model()
            train_loader, val_loader = get_train_loaders(train_dir, val_dir, batch_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience, warmup_steps)
            trainer.train(num_epochs)
            st.write("Model Training Completed")

with tab2:
    st.header("Model Evaluation")
    st.write("This is the model evaluation page")
    st.write("Please select the model and test directory")

    with st.form(key="model_evaluation"):
        model_list = os.listdir('C:\\uoft\\1517\\project\\project\\Streamlit\\ckpts')
        model_dir = st.selectbox("Model", model_list, index=None,placeholder="Select Model...")
        test_dir= 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\test'
        batch_size = st.number_input("Batch Size", 0)
        model = Model()
        submit=st.form_submit_button("Evaluate Model")

        if submit:
            model.load_state_dict(torch.load(model_dir))
            model.eval()
            test_loader = get_test_loader(test_dir, batch_size)
            criterion = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = Trainer(model, None, test_loader, criterion, None, None, device)
            accuracy = trainer.evaluate()
            st.write(f"Model Accuracy: {accuracy}")

