import streamlit as st
import gc
import pandas as pd
import numpy as np
import torch
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SET THE PATHS (The Fix) ---
# We define script_dir here so the function below can see it
script_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(script_dir, 'career_archetypes.pkl')
map_path = os.path.join(script_dir, 'archetype_mapping.parquet')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Career Pivot Engine", layout="centered")

@st.cache_resource
# --- 2. LOAD THE ENGINE ---

def get_engine():
    # Load the Brain
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)

    # Load the Model
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    # Load the Course Map
    course_df = pd.read_parquet(map_path).sample(n=15000) 

    gc.collect()
   
    return brain, course_df, model


# --- 2. THE UI ---
st.title("🎯 Career Pivot & Gap Analysis")
st.write("Paste your resume and a target job description to see what you're missing.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Your Resume", height=250, placeholder="Paste resume content here...")
with col2:
    jd_text = st.text_area("Target Job Description", height=250, placeholder="Paste the job you want here...")

if st.button("Analyze Career Gap"):
    if jd_text: # We only strictly need the JD to proceed
        with st.spinner("Analyzing career path..."):
            # 1. Vectorize the Job Description
            # Call the loader
            brain, course_df, model = get_engine()
            jd_vec = model.encode([jd_text], convert_to_tensor=True)
            
            # 2. Determine the "Target Vector"
            if resume_text.strip():
                # If resume exists, calculate the GAP
                res_vec = model.encode([resume_text], convert_to_tensor=True)
                target_vector = jd_vec - res_vec
                st.info("💡 Analysis Mode: Resume Gap (Targeting what you're missing)")
            else:
                # If no resume, target the JD directly
                gc.collect()
                target_vector = jd_vec
                st.info("💡 Analysis Mode: Direct Job Match (Showing core requirements)")

            # 3. Project to PCA space
            pca_v = torch.tensor(brain['pca_v'], dtype=torch.float32)
            pca_mean = torch.tensor(brain['pca_mean'], dtype=torch.float32)
            
            centered_target = target_vector - pca_mean
            # Ensure it is float32 and flattened to a 2D array with 1 row
            reduced_target = torch.mm(centered_target, pca_v).detach().cpu().numpy().astype(np.float32)
            
            # --- THE SAFETY CHECK ---
            # kmeans.predict usually wants a 2D array: [[val, val, val...]]
            if reduced_target.ndim == 1:
                reduced_target = reduced_target.reshape(1, -1)

            # --- 4. Find Archetype (Manual Calculation to bypass Buffer Error) ---
            # Get the centroids from your KMeans model
            centroids = brain['kmeans_model'].cluster_centers_ # These are already float32
            
            # Calculate Euclidean Distance from our target to every centroid
            # Distance = sqrt( sum( (a - b)^2 ) )
            diff = centroids - reduced_target
            dist_sq = np.sum(np.square(diff), axis=1)
            closest_index = np.argmin(dist_sq)
            
            cluster_id = str(closest_index)
            recommendations = course_df[course_df['archetype_id'] == cluster_id].to_dict('records')
            
            # E. Score and Rank
            scored = []
            for item in recommendations:
                # Convert both to float64 just for the similarity calculation
                item_vec = np.array([item['embedding']], dtype=np.float64)
                target_vec_64 = reduced_target.astype(np.float64).reshape(1, -1)
                
                sim = cosine_similarity(target_vec_64, item_vec)[0][0]
                
                item_copy = item.copy()
                item_copy['score'] = float(sim)
                scored.append(item_copy)
            
            scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:5]
            
            # F. DISPLAY RESULTS
            st.success(f"Gap Analysis Complete! Matched to Career Archetype: {cluster_id}")
            st.subheader("Top Resources to Bridge Your Gap:")
            
            for i, rec in enumerate(scored, 1):
                with st.container():
                    st.markdown(f"**{i}. {rec['title']}**")
                    score_val = float(rec['score'])
                    st.progress(min(max(score_val, 0.0), 1.0), text=f"{score_val:.1%} Match")
                    st.write(f"[View Resource]({rec['url']})")
                    st.divider()
    else:
        st.warning("Please paste at least a job description.")
