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
st.set_page_config(page_title="Career Pivot Engine", layout="centered", page_icon="🎯")

# --- 2. THE OPTIMIZED JINA ENGINE ---
@st.cache_resource
def get_engine():
    # Load the Brain (Now with 350-D PCA artifacts)
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)

    # Load Jina in Half-Precision (Crucial for 1GB RAM limit)
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en", 
        trust_remote_code=True, 
        device='cpu',
        model_kwargs={"torch_dtype": torch.float16} 
    )
    
    # Load the Course Map (The 350-D Parquet)
    course_df = pd.read_parquet(map_path) 
    
    gc.collect()
    return brain, course_df, model

# --- 3. THE UI ---
st.title("🎯 Career Pivot & Gap Analysis")
st.markdown("""
    ### Bridge the gap between where you are and where you want to be.
    Paste your **Resume** and a **Job Description** to find the best Udemy courses to level up.
""")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("📄 Your Resume", height=250, placeholder="Paste resume content here...")
with col2:
    jd_text = st.text_area("💼 Target Job Description", height=250, placeholder="Paste the job you want here...")

if st.button("Analyze Career Gap", type="primary"):
    if jd_text:
        with st.spinner("Crunching data with Jina-AI..."):
            # A. Load heavy resources
            brain, course_df, model = get_engine()
            
            # B. Vectorize (Returns 768-D)
            jd_vec = model.encode([jd_text], convert_to_tensor=True)
            
            # C. Determine Target Vector
            if resume_text.strip():
                res_vec = model.encode([resume_text], convert_to_tensor=True)
                # The "Gap" is what the JD has that the Resume doesn't
                target_vector = jd_vec - res_vec
                st.info("💡 **Analysis Mode:** Resume Gap (Targeting missing skills)")
            else:
                target_vector = jd_vec
                st.info("💡 **Analysis Mode:** Direct Job Match (Showing core requirements)")

            # D. PCA PROJECTION (768-D -> 350-D)
            # Use the brain components you just generated locally
            pca_v = torch.tensor(brain['pca_v'], dtype=torch.float32)
            pca_mean = torch.tensor(brain['pca_mean'], dtype=torch.float32)
            
            centered = target_vector - pca_mean
            # Matrix multiplication to project into the 350-D space
            reduced_target = torch.mm(centered, pca_v).detach().cpu().numpy().reshape(1, -1)

            # E. FIND ARCHETYPE (Using the 768-D centroids in your brain)
            centroids_768 = brain['centroids']
            target_768 = target_vector.detach().cpu().numpy().reshape(1, -1)
            
            # Manual Euclidean Distance calculation
            diff = centroids_768 - target_768
            dist_sq = np.sum(np.square(diff), axis=1)
            closest_index = np.argmin(dist_sq)
            cluster_id = str(closest_index)

            # F. FILTER & RANK COURSES
            # Filter the parquet by the matched archetype
            recommendations = course_df[course_df['archetype_id'] == cluster_id].to_dict('records')
            
            scored = []
            for item in recommendations:
                # Use the 350-D embedding stored in the Parquet
                item_emb = np.array(item['embedding']).reshape(1, -1)
                sim = cosine_similarity(reduced_target, item_emb)[0][0]
                
                item_copy = item.copy()
                item_copy['score'] = float(sim)
                scored.append(item_copy)
            
            # Rank by top 5 matches
            scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:5]

            # G. DISPLAY RESULTS
            st.success(f"Gap Analysis Complete! Matched to Career Archetype: **{cluster_id}**")
            
            if scored:
                st.subheader("Recommended Resources to Bridge the Gap:")
                for i, rec in enumerate(scored, 1):
                    with st.container():
                        st.markdown(f"**{i}. {rec['title']}**")
                        score_val = float(rec['score'])
                        # Clip progress bar to 0-1 range
                        st.progress(min(max(score_val, 0.0), 1.0), text=f"{score_val:.1%} Match")
                        st.write(f"🔗 [View Course on Udemy]({rec['url']})")
                        st.divider()
            else:
                st.warning("We found your career path, but didn't find specific courses in this sample. Try broadening your JD.")
            
            # H. Cleanup RAM
            gc.collect()
    else:
        st.warning("Please paste at least a Job Description to begin.")
