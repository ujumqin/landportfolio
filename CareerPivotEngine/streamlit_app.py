import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import gc
import time
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SET THE PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(script_dir, 'career_archetypes.pkl')
map_path = os.path.join(script_dir, 'archetype_mapping.parquet')

# --- HUGGING FACE API SETUP (Hub-Style URL) ---
# This bypasses the general router and goes to the specific model's API endpoint
API_URL = "https://router.huggingface.co/hf-inference/models/jinaai/jina-embeddings-v2-base-en"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query_jina(text):
    """The most stable payload format for Jina v2."""
    payload = {
        "inputs": text, # Try it without the brackets first for this specific endpoint
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # DEBUG: This will show in your Streamlit logs if it fails again
        if response.status_code != 200:
            return {"error": f"Code {response.status_code}: {response.text[:100]}"}
            
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- 2. LOAD DATA ---
@st.cache_resource
def load_assets():
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)
    course_df = pd.read_parquet(map_path) 
    return brain, course_df

# --- 3. THE UI ---
st.set_page_config(page_title="Career Pivot Engine", layout="centered", page_icon="🎯")
st.title("🎯 Career Pivot & Gap Analysis")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("📄 Your Resume", height=250, placeholder="Paste resume...")
with col2:
    jd_text = st.text_area("💼 Job Description", height=250, placeholder="Paste JD...")

if st.button("Analyze Career Gap", type="primary"):
    if jd_text:
        with st.spinner("Connecting to Jina-AI Cloud..."):
            brain, course_df = load_assets()
            
            # 1. Get Embeddings
            jd_resp = query_jina(jd_text.strip())
            
            # Check if the response is valid data (a list) or an error (a dict)
            if isinstance(jd_resp, list):
                # Flatten the nested list if necessary
                jd_array = np.array(jd_resp)
                if jd_array.ndim > 2:
                    jd_array = jd_array.squeeze() # Remove extra dims
                jd_vec = jd_array.astype(np.float32).reshape(1, -1)
                
                if resume_text.strip():
                    res_resp = query_jina(resume_text.strip())
                    if isinstance(res_resp, list):
                        res_vec = np.array(res_resp).astype(np.float32).reshape(1, -1)
                        target_vector = jd_vec - res_vec
                        st.info("💡 **Analysis Mode:** Resume Gap")
                    else:
                        st.error("Could not process resume via API.")
                        st.stop()
                else:
                    target_vector = jd_vec
                    st.info("💡 **Analysis Mode:** Direct Match")

                # 2. PCA PROJECTION (768 -> 350)
                pca_v = np.array(brain['pca_v'], dtype=np.float32)
                pca_mean = np.array(brain['pca_mean'], dtype=np.float32)
                reduced_target = (target_vector - pca_mean) @ pca_v

                # 3. ARCHETYPE MATCH (Using 768-D Centroids)
                centroids = np.array(brain['centroids'], dtype=np.float32)
                dist_sq = np.sum(np.square(centroids - target_vector), axis=1)
                cluster_id = str(np.argmin(dist_sq))

                # 4. FILTER & RANK
                recs = course_df[course_df['archetype_id'] == cluster_id].to_dict('records')
                
                scored = []
                for item in recs:
                    item_emb = np.array(item['embedding']).reshape(1, -1)
                    sim = cosine_similarity(reduced_target, item_emb)[0][0]
                    item['score'] = float(sim)
                    scored.append(item)
                
                scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:5]

                # 5. DISPLAY
                st.success(f"Matched to Career Archetype: **{cluster_id}**")
                for i, rec in enumerate(scored, 1):
                    with st.container():
                        st.markdown(f"**{i}. {rec['title']}**")
                        score_val = rec['score']
                        st.progress(min(max(score_val, 0.0), 1.0), text=f"{score_val:.1%} Match")
                        st.write(f"[View Course]({rec['url']})")
                        st.divider()
            
            elif isinstance(jd_resp, dict) and "estimated_time" in jd_resp:
                st.warning(f"Jina-AI is waking up. Please wait about {int(jd_resp['estimated_time'])} seconds and try again.")
            else:
                st.error(f"API Error: {jd_resp}")
            
            gc.collect()
    else:
        st.warning("Please paste at least a Job Description.")
