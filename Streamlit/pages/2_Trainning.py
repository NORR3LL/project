import streamlit as st
import torch.nn as nn
import torch.optim as optim
import torch
import os
import time
import pandas as pd

from scripts.trainners import Trainer, get_train_loaders, get_test_loader, evaluate_model
from scripts.models import Model, Baseline

# Page Configuration
st.set_page_config(
    page_title="Model Training and Validation",
    page_icon="📈",
    layout="wide",
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

# Tabs for navigation
tab1, tab2 = st.tabs(["📊 Model Training", "📈 Model Evaluation"])

# ===================== Model Training Tab ===================== #
with tab1:
    st.markdown('<div class="header-style">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header-style">Set up training parameters and start training the model.</div>', unsafe_allow_html=True)
    
    st.write("The GPU is available: ", torch.cuda.is_available())
    
    with st.form(key="model_training"):
        # Input fields for training configuration
        st.markdown("### 🔧 Training Configuration")
        train_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\train'
        val_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\val'
        csv_output_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Plots'
        ckpt_output_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts'

        batch_size = st.number_input("Batch Size", value=1)
        num_epochs = st.number_input("Number of Epochs", value=1)
        learning_rate = st.number_input("Learning Rate", value=0.0001, step=0.0001, format="%.4f")
        num_classes = st.number_input("Number of Classes", value=5)
        patience = st.number_input("Patience", value=0)
        warmup_steps = st.number_input("Warmup Steps", value=0)

        # Initialize models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(num_classes).to(device)
        baseline_model = Baseline(num_classes).to(device)

        submit = st.form_submit_button("🚀 Train Model")
        
        if submit:
            st.info(f"Model Training Started on {device}.")
            start_time = time.time()
            date = time.strftime("%Y-%m-%d-%H-%M-%S")

            # Load data
            train_loader, val_loader = get_train_loaders(train_dir, val_dir, batch_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # Train models
            baseline_model_trainer = Trainer(baseline_model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience, warmup_steps, csv_output_dir)
            baseline_model_trainer.train(num_epochs)

            model_trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience, warmup_steps, csv_output_dir)
            model_trainer.train(num_epochs)

            # Save models and metrics
            baseline_model_name = f"baseline_{date}.pth"
            model_name = f"model_{date}.pth"
            baseline_ckpt_dir = os.path.join(ckpt_output_dir, baseline_model_name)
            model_ckpt_dir = os.path.join(ckpt_output_dir, model_name)

            torch.save(baseline_model_trainer.model.state_dict(), baseline_ckpt_dir)
            torch.save(model_trainer.model.state_dict(), model_ckpt_dir)

            baseline_model_trainer.save_metrics_to_csv(csv_filename=baseline_model_name)
            model_trainer.save_metrics_to_csv(csv_filename=model_name)

            # Display results
            st.success("Model Training Completed!")
            st.write(f"Baseline Model Checkpoint: `{baseline_ckpt_dir}`")
            st.write(f"Model Checkpoint: `{model_ckpt_dir}`")
            end_time = time.time()
            duration = (end_time - start_time) / 60
            st.info(f"Training Duration: {duration:.2f} minutes.")

# ===================== Model Evaluation Tab ===================== #
with tab2:
    st.markdown('<div class="header-style">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header-style">Select a trained model and evaluate it on the test set.</div>', unsafe_allow_html=True)
    
    with st.form(key="model_evaluation"):
        # Dropdowns for model selection
        st.markdown("### 🔍 Evaluation Configuration")
        model_list = [f for f in os.listdir('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts') if 'resnet' in f and f.endswith('.pth')]
        model_selection = st.selectbox("Model", model_list, index=None, placeholder="Select Model...")

        baseline_model_list = [f for f in os.listdir('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts') if 'baseline' in f and f.endswith('.pth')]
        baseline_model_selection = st.selectbox("Baseline Model", baseline_model_list, index=None, placeholder="Select Baseline Model...")

        test_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Gold\\test'
        csv_output_dir = 'C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\Plots'

        batch_size = st.number_input("Batch Size", value=32)
        submit = st.form_submit_button("📊 Evaluate Model")

        if submit:
            start_time = time.time()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load models
            model = Model(num_classes=5).to(device)
            model_path = os.path.join('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts', model_selection)
            model.load_state_dict(torch.load(model_path, map_location=device))

            baseline_model = Baseline(num_classes=5).to(device)
            baseline_model_path = os.path.join('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts', baseline_model_selection)
            baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))

            # Evaluate
            st.info("Model Evaluation Started.")
            test_loader = get_test_loader(test_dir, batch_size)

            # Load evaluation results
            test_results = pd.read_csv(os.path.join(csv_output_dir, 'test_2024-11-24_00-44-58.csv'))
            st.markdown("### 📄 Test Results")
            st.write(test_results)

            # Plot results
            baseline_stats = pd.read_csv(os.path.join(csv_output_dir, baseline_model_selection.replace('.pth', '.csv')))
            model_stats = pd.read_csv(os.path.join(csv_output_dir, model_selection.replace('.pth', '.csv')))
            

            # # evaluate model
            # criterion = nn.CrossEntropyLoss()
            # baseline_test_loss,baseline_test_acc = evaluate_model(baseline_model, test_loader, criterion, device)
            # st.write("Baseline Model Evaluation Completed")
            # model_test_loss,model_test_acc = evaluate_model(model, test_loader, criterion, device)
            # st.write("Model Evaluation Completed")

            # st.write(f"Baseline Model Test Accuracy: {baseline_test_acc:.4f}")
            # st.write(f"Model Test Accuracy: {model_test_acc:.4f}")

            st.markdown("### 📈 Baseline Model Training and Validation")
            st.line_chart(baseline_stats[['Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']])

            st.markdown("### 📈 Model Training and Validation")
            st.line_chart(model_stats[['Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']])

            end_time = time.time()
            duration = (end_time - start_time) / 60
            st.info(f"Evaluation Duration: {duration:.2f} minutes.")

            
