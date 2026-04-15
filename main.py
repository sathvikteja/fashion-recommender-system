import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fashion Recommender", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 30px;
        color: #facc15;
    }

    .stButton>button {
        background-color: #38bdf8;
        color: black;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title"> 👕 Fashion Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find similar fashion items instantly using AI</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ---------------- MODEL ----------------
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ---------------- SAVE FILE ----------------
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        file_path = os.path.join('uploads', uploaded_file.name)

        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ---------------- FEATURE EXTRACTION ----------------
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# ---------------- RECOMMEND ----------------
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# ---------------- UPLOAD ----------------
st.markdown("### 📤 Upload a fashion image")
uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'])

if uploaded_file is not None:

    file_path = save_uploaded_file(uploaded_file)

    if file_path:

        col1, col2 = st.columns([1,2])

        # Uploaded Image
        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            display_image = Image.open(uploaded_file)
            st.image(display_image, width=250)

        # Processing
        with col2:
            with st.spinner("🔍 Finding similar items..."):
                features = feature_extraction(file_path, model)
                indices = recommend(features, feature_list)

            st.success("✅ Recommendations ready!")

        # ---------------- RESULTS ----------------
        st.markdown('<div class="section-title">🛍️ Recommended Products</div>', unsafe_allow_html=True)

        cols = st.columns([1,1,1,1,1])

        for i in range(5):
            with cols[i]:
                st.image(filenames[indices[0][i]], width=150)

    else:
        st.error("❌ Error saving file")