import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import gc

# --- 1. SET THE PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(script_dir, 'career_archetypes.pkl')
map_path = os.path.join(script_dir, 'archetype_mapping.parquet')

# --- HUGGING FACE API SETUP ---
API_URL = "https://api-inference.huggingface.co/models/jinaai/jina-embeddings-v2-base-en"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query_jina(text_list):
    response = requests.post(API_URL, headers=headers, json={"inputs": text_list})
    return response.json()

# --- 2. LOAD THE ENGINE (Lighter Version) ---
@st.cache_resource
def load_data():
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)
    course_df = pd.read_parquet(map_path) 
    return brain, course_df

# --- 3. THE UI ---
st.set_page_config(page_title="Career Pivot Engine", layout="centered")
st.title("🎯 Career Pivot & Gap Analysis")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Your Resume", height=250)
with col2:
    jd_text = st.text_area("Target Job Description", height=250)

if st.button("Analyze Career Gap"):
    if jd_text:
        with st.spinner("Talking to Jina-AI..."):
            brain, course_df = load_data()
            
            # 1. Get Embeddings via API (Zero RAM usage!)
            # Jina API usually returns a list of lists
            jd_vec_list = query_jina([jd_text])
            jd_vec = np.array(jd_vec_list).astype(np.float32)
            
            if resume_text.strip():
                res_vec_list = query_jina([resume_text])
                res_vec = np.array(res_vec_list).astype(np.float32)
                target_vector = jd_vec - res_vec
            else:
                target_vector = jd_vec

            # 2. PCA PROJECTION (Matches your 350-D save)
            pca_v = np.array(brain['pca_v'], dtype=np.float32)
            pca_mean = np.array(brain['pca_mean'], dtype=np.float32)
            
            reduced_target = (target_vector - pca_mean) @ pca_v

            # 3. ARCHETYPE MATCH
            centroids = np.array(brain['centroids'], dtype=np.float32)
            dist_sq = np.sum(np.square(centroids - target_vector), axis=1)
            cluster_id = str(np.argmin(dist_sq))

            # 4. FILTER & DISPLAY
            recs = course_df[course_df['archetype_id'] == cluster_id].to_dict('records')
            # ... [Ranking and Display logic remains same as before] ...
            st.success(f"Matched to Archetype: {cluster_id}")
            # [Insert your existing scoring/display loop here]
