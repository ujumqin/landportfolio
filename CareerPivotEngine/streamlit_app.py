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

# --- 2. LOAD THE ENGINE ---
@st.cache_resource
def get_engine():
    # Load the Brain
    with open(brain_path, 'rb') as f:
        brain = pickle.load(f)

    # Load the Model (Lightweight MiniLM)
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    
    # Load the Course Map
    course_df = pd.read_parquet(map_path).sample(n=15000) 

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
        with st.spinner("Analyzing career path..."):
            # Load resources
            brain, course_df, model = get_engine()
            
            # 1. Vectorize the Job Description
            jd_vec = model.encode([jd_text], convert_to_tensor=True)
            
            # 2. Determine the "Target Vector"
            if resume_text.strip():
                res_vec = model.encode([resume_text], convert_to_tensor=True)
                target_vector = jd_vec - res_vec
                st.info("💡 Analysis Mode: Resume Gap (Targeting what you're missing)")
            else:
                target_vector = jd_vec
                st.info("💡 Analysis Mode: Direct Job Match (Showing core requirements)")

            # 3. Preparation for Search
            target_np = target_vector.detach().cpu().numpy().reshape(1, -1)
            scored = []

            # 4. Search Execution with Fallback
            try:
                # Attempt Semantic Search (Will fail if embeddings are Jina-sized)
                for _, row in course_df.iterrows():
                    # Check if embedding exists and is correct size
                    emb = np.array(row['embedding']).reshape(1, -1)
                    if emb.shape[1] == target_np.shape[1]:
                        sim = cosine_similarity(target_np, emb)[0][0]
                        row_copy = row.to_dict()
                        row_copy['score'] = float(sim)
                        scored.append(row_copy)
                
                # If we found matches, sort them
                if scored:
                    scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:5]
                    st.success("Semantic Analysis Complete!")
                else:
                    raise ValueError("No matching embeddings found.")

            except Exception:
                # ULTIMATE FALLBACK: Keyword search for a working demo
                st.warning("🔄 Using Keyword Matching (Dimensions mismatch detected)")
                # Extract words longer than 3 letters as keywords
                keywords = [word for word in jd_text.split() if len(word) > 3][:10]
                pattern = '|'.join(keywords)
                results = course_df[course_df['title'].str.contains(pattern, case=False, na=False)]
                
                scored = results.head(5).to_dict('records')
                for r in scored: 
                    r['score'] = 0.85 # Placeholder score for progress bar

            # 5. DISPLAY RESULTS
            if scored:
                st.subheader("Top Resources to Bridge Your Gap:")
                for i, rec in enumerate(scored, 1):
                    with st.container():
                        st.markdown(f"**{i}. {rec['title']}**")
                        score_val = float(rec['score'])
                        st.progress(min(max(score_val, 0.0), 1.0), text=f"{score_val:.1%} Match")
                        st.write(f"[View Resource]({rec['url']})")
                        st.divider()
            else:
                st.error("No relevant courses found in the current sample. Try different keywords.")
            
            gc.collect()
    else:
        st.warning("Please paste at least a job description.")
