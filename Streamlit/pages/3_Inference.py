import streamlit as st
import os
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms
from scripts.models import Model

# Set page configuration
st.set_page_config(
    page_title="Inference",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a header with a styled title
st.markdown(
    """
    <style>
    .header-style {
        font-size:36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .sub-header-style {
        font-size:18px;
        color: #555;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header-style">Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-style">Upload an image and select a model to evaluate predictions.</div>', unsafe_allow_html=True)

# Add an image upload section
st.sidebar.title("🔧 Configuration")
st.sidebar.info("Upload an image and choose a model from the dropdown below.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale if necessary
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

with st.form(key="model_evaluation"):
    st.markdown("### 📤 Upload Image")
    image = st.file_uploader("Choose an image to evaluate", type=["jpg", "png", "jpeg"])
    
    st.markdown("### 🔍 Select Model")
    model_list = [
        f for f in os.listdir('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts') 
        if 'resnet' in f and f.endswith('.pth')
    ]
    model_selection = st.selectbox("Model", model_list, index=None, placeholder="Select Model...")
    
    submit = st.form_submit_button("🔍 Evaluate Model")

if submit:
    if image and model_selection:
        st.markdown("### 📊 Model Evaluation in Progress")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Model(num_classes=5)
        model_path = os.path.join('C:\\uoft\\1517\\project\\project\\Streamlit\\outputs\\ckpts', model_selection)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Open and preprocess image
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Move image to device
        img = img.to(device)
        
        class_labels = ['Lung Opacity', 'Normal', 'Bacterial Pneumonia', 'Covid-19', 'Viral Pneumonia']

        # Predict
        with torch.no_grad():
            prediction = model(img)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
   
        # Get the predicted class index
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        probabilities = probabilities.cpu().numpy()
        probabilities_df = pd.DataFrame(probabilities, columns=class_labels)

        # Map the index to a class label
        predicted_class = class_labels[predicted_class_index]

        # Display result
        st.markdown("### 🎯 Prediction Results")
        st.markdown(f"**Predicted Class:** `{predicted_class}`")

        st.dataframe(probabilities_df.style.highlight_max(axis=1, color='lightgreen'))
    else:
        st.warning("Please upload an image and select a model.")
