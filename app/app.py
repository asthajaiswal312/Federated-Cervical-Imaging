import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="CerviImagingDiag: Privacy-Preserved FL & XAI",
    layout="centered"
)

st.title("🔬 CerviImagingDiag: Federated Learning & XAI Platform")
st.markdown("A lightweight, privacy-preserving AI framework for early-stage cervical screening on edge/consumer devices[cite: 1].")

# 2. Define Class Labels (Must match your SipakMed training order)
CLASSES = [
    'im_Dyskeratotic', 
    'im_Koilocytotic', 
    'im_Metaplastic', 
    'im_Parabasal', 
    'im_Superficial-Intermediate'
]

# 3. Medical Knowledge Mapping for XAI Generation
CLASS_MEDICAL_PROFILES = {
    'im_Dyskeratotic': {
        'grade': 'Abnormal / High-Grade Squamous Intraepithelial Precursor',
        'description': 'Shows distinct cellular keratinization and nuclear atypia.',
        'clinical_advice': 'Indicates potential high-grade precancerous changes. Immediate consultation with a gynecological oncologist and a colposcopy-guided biopsy are strongly recommended.'
    },
    'im_Koilocytotic': {
        'grade': 'Abnormal / Low-Grade Precancerous Change (HPV-related)',
        'description': 'Exhibits characteristic koilocytotic halos, indicative of Human Papillomavirus (HPV) infection and mild dysplasia.',
        'clinical_advice': 'Low-grade lesion associated with viral changes. Routine follow-up screening within 6 months and further HPV-typing tests are advised.'
    },
    'im_Metaplastic': {
        'grade': 'Benign / Metaplastic Transformation',
        'description': 'Represents normal tissue healing or cellular adaptation within the transformation zone of the cervix.',
        'clinical_advice': 'Benign finding with favorable prognosis. Standard annual routine cervical screening is recommended.'
    },
    'im_Parabasal': {
        'grade': 'Benign / Immature Squamous Cells',
        'description': 'Normal deep epithelial cells occasionally observed in atrophic or regular physiological smears.',
        'clinical_advice': 'Normal physiological presentation. No immediate clinical intervention required.'
    },
    'im_Superficial-Intermediate': {
        'grade': 'Normal / Healthy Mature Cells',
        'description': 'Normal mature squamous epithelial cells reflecting healthy cellular shedding.',
        'clinical_advice': 'Normal finding. Maintain regular age-appropriate preventive screenings.'
    }
}

# 4. Load the Model (Cached for performance)
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    
    # Load your saved federated checkpoint weights
    try:
        model.load_state_dict(torch.load("federated_cervical_model.pth", map_location=device))
    except FileNotFoundError:
        st.error("Model checkpoint 'federated_cervical_model.pth' not found in directory!")
    
    model.to(device)
    model.eval()
    return model, device

model, device = load_trained_model()

# 5. Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. Sidebar Controls & File Uploader
st.sidebar.header("Patient Portal")
patient_id = st.sidebar.text_input("Enter Patient ID", value="P-2026-01")

st.header("Upload Cervical Cell Image")
uploaded_file = st.file_uploader("Choose a cell image (.bmp or .jpg)...", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cervical Cytology Sample", use_column_width=True)
    
    if st.button("Run Federated AI Diagnostics"):
        with st.spinner("Analyzing cell sample across federated nodes..."):
            # Preprocess and predict
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            pred_class_name = CLASSES[predicted_idx.item()]
            conf_score = confidence.item() * 100
            
            profile = CLASS_MEDICAL_PROFILES.get(pred_class_name, {})
            
        # 7. Render Diagnostic Results Dashboard
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        col1.metric("Predicted Class", pred_class_name)
        col2.metric("Confidence Score", f"{conf_score:.2f}%")
        
        st.markdown("---")
        st.subheader("📋 AI-Generated Clinical Report")
        
        report_markdown = f"""
        **Patient ID:** {patient_id}  
        **Primary Classification:** `{pred_class_name}`  
        **Diagnostic Category:** {profile.get('grade')}  

        ### 1. Summary of Findings
        The federated edge model detected features matching **{profile.get('grade')}** with a confidence of **{conf_score:.2f}%**. 
        
        ### 2. Clinical Interpretation
        *Cellular Observations:* {profile.get('description')}  
        
        ### 3. Recommended Next Steps
        {profile.get('clinical_advice')}
        """
        
        st.markdown(report_markdown)
        
        # Download button for report
        st.download_button(
            label="Download Clinical Report as Text",
            data=report_markdown,
            file_name=f"Report_{patient_id}.txt",
            mime="text/plain"
        )