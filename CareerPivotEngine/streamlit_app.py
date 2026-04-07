import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SET THE PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(script_dir, 'career_archetypes.pkl')
map_path = os.path.join(script_dir, 'archetype_mapping.parquet')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Career Pivot Engine", layout="centered")

# --- 2. THE OPTIMIZED JINA ENGINE ---
@st.cache_resource
def get_engine():
    # Load the Brain (Now with 350-D PCA artifacts)
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)

    # Load Jina (Switching back from MiniLM)
    # We use trust_remote_code=True for Jina's specific architecture
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device='cpu')
    
    # Load the Course Map (The 350-D Parquet)
    course_df = pd.read_parquet(map_path) 
    
    gc.collect()
    return brain, course_df, model

# --- 3. THE UI ---
st.title("🎯 Career Pivot & Gap Analysis")
st.write("Paste your resume and a target job description to see what you're missing.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Your Resume", height=250, placeholder="Paste resume content here...")
with col2:
    jd_text = st.text_area("Target Job Description", height=250, placeholder="Paste the job you want here...")

if st.button("Analyze Career Gap"):
    if jd_text:
        with st.spinner("Initializing Jina-AI Engine... (350-D Mode)"):
            # Load heavy resources
            brain, course_df, model = get_engine()
            
            # 1. Vectorize (Returns 768-D)
            jd_vec = model.encode([jd_text], convert_to_tensor=True)
            
            # 2. Determine Target
            if resume_text.strip():
                res_vec = model.encode([resume_text], convert_to_tensor=True)
                target_vector = jd_vec - res_vec
                st.info("💡 Analysis Mode: Resume Gap")
            else:
                target_vector = jd_vec
                st.info("💡 Analysis Mode: Direct Job Match")

            # 3. PCA PROJECTION (The 768 -> 350 Magic)
            # Ensure these are tensors for fast math
            pca_v = torch.tensor(brain['pca_v'], dtype=torch.float32)
            pca_mean = torch.tensor(brain['pca_mean'], dtype=torch.float32)
            
            centered = target_vector - pca_mean
            # Multiply 768-D target by the 768x350 PCA matrix
            reduced_target = torch.mm(centered, pca_v).detach().cpu().numpy().reshape(1, -1)

            # 4. FIND CLOSEST CENTROID (Manual Math)
            centroids = brain['centroids'] # These are stored in 768-D or 350-D depending on your save
            # If your centroids in the pkl are 768-D, we use target_vector
            # If your centroids in the pkl are 350-D, we use reduced_target
            # Based on your save script, they are 768-D:
            target_768 = target_vector.detach().cpu().numpy().reshape(1, -1)
            
            diff = centroids - target_768
            dist_sq = np.sum(np.square(diff), axis=1)
            closest_index = np.argmin(dist_sq)
            cluster_id = str(closest_index)

            # 5. FILTER & SCORE
            recommendations = course_df[course_df['archetype_id'] == cluster_id].to_dict('records')
            
            scored = []
            for item in recommendations:
                # We use the 350-D embeddings from the Parquet for the final ranking
                item_emb = np.array(item['embedding']).reshape(1, -1)
                sim = cosine_similarity(reduced_target, item_emb)[0][0]
                
                item_copy = item.copy()
                item_copy['score'] = float(sim)
                scored.append(item_copy)
            
            scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:5]

            # 6. DISPLAY RESULTS
            st.success(f"Gap Analysis Complete! Archetype: {cluster_id}")
            if scored:
                for i, rec in enumerate(scored, 1):
                    with st.container():
                        st.markdown(f"**{i}. {rec['title']}**")
                        score_val = float(rec['score'])
                        st.progress(min(max(score_val, 0.0), 1.0), text=f"{score_val:.1%} Match")
                        st.write(f"[View Resource]({rec['url']})")
                        st.divider()
            else:
                st.warning("Found the archetype, but no courses were in this specific sample.")
            
            gc.collect()
    else:
        st.warning("Please paste at least a job description.")
