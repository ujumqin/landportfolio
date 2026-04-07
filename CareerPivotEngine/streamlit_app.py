import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# --- 1. SET THE PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(script_dir, 'career_archetypes.pkl')
map_path = os.path.join(script_dir, 'archetype_mapping.parquet')

# --- HUGGING FACE API SETUP ---
# Initializing the client once
client = InferenceClient(
    model="jinaai/jina-embeddings-v2-base-en",
    token=st.secrets["HF_TOKEN"]
)

def query_jina(text):
    """Uses the dedicated feature_extraction method to avoid StopIteration."""
    try:
        # We explicitly call feature_extraction and turn off streaming
        # to ensure we get the full vector at once.
        embedding = client.feature_extraction(
            text.strip(),
            model="jinaai/jina-embeddings-v2-base-en"
        )
        
        # Convert the result (which is a numpy-like array) to a standard list
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)
        
    except Exception as e:
        # If it still fails, we check if it's a known 'Loading' state
        error_msg = str(e)
        if "currently loading" in error_msg.lower():
            return {"estimated_time": 20} # Triggers your existing warning UI
        return {"error": f"Cloud Error: {error_msg if error_msg else 'Model Wake-up Lag'}"}
        
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
