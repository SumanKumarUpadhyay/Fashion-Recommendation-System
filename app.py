import pickle
import numpy as np
from PIL import Image
import streamlit as st
from src.feature import extract_features
from src.recommend import recommend

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stApp"] {
    background: #111114 !important;
    color: #e4e4e7;
    font-family: 'DM Sans', sans-serif;
    overflow-x: hidden;
}

[data-testid="stToolbar"],
[data-testid="manage-app-button"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
footer, header, .stDeployButton,
div.stAlert { display: none !important; }

.block-container {
    padding: 1.5rem 2rem 1rem !important;
    max-width: 1320px !important;
    margin: 0 auto;
}

.frs-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid rgba(255,182,193,0.12);
}
.frs-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(120deg, #f9a8b8 0%, #ffd6de 60%, #ffb6c1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.frs-sub {
    font-size: 0.8rem;
    color: #71717a;
    margin-top: 3px;
    letter-spacing: 0.03em;
}

.frs-card {
    background: linear-gradient(160deg, #18181b 0%, #141417 100%);
    border: 1px solid rgba(255,182,193,0.1);
    border-radius: 18px;
    padding: 1.1rem 1.4rem 1.5rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.03);
}

[data-testid="stFileUploader"] {
    background: rgba(255,182,193,0.03) !important;
    border: 1.5px dashed rgba(255,182,193,0.2) !important;
    border-radius: 10px !important;
    padding: 0.2rem 1rem 0.5rem !important;
    margin-bottom: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,182,193,0.4) !important;
}
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #71717a !important;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(249,168,184,0.1) !important;
    border: 1px solid rgba(249,168,184,0.28) !important;
    color: #f9a8b8 !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    padding: 3px 14px !important;
}

.sec-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f9a8b8;
    margin-bottom: 0.55rem;
}

[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,182,193,0.14) !important;
    width: 100% !important;
    object-fit: cover;
    max-height: 340px;
    min-height: 220px;
}

.match-badge {
    margin-top: 8px;
    background: rgba(249,168,184,0.07);
    border: 1px solid rgba(249,168,184,0.18);
    border-radius: 8px;
    padding: 5px 11px;
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
}
.match-rank { color: #52525b; font-weight: 500; }
.match-pct  { color: #f9a8b8; font-weight: 700; letter-spacing: 0.02em; }

.ph-card {
    background: rgba(255,182,193,0.02);
    border: 1px dashed rgba(255,182,193,0.09);
    border-radius: 12px;
    height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #3f3f46;
    font-size: 0.7rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    gap: 8px;
}

.v-div {
    width: 1px;
    background: linear-gradient(to bottom,
        transparent,
        rgba(255,182,193,0.15) 20%,
        rgba(255,182,193,0.15) 80%,
        transparent);
    min-height: 260px;
    margin: 1.5rem auto 0;
}

[data-testid="stSpinner"] p { color: #f9a8b8 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    with open("models/image_features.pkl", "rb") as f:
        raw = pickle.load(f)
    feature_list = np.vstack(raw).astype(np.float32)   # (N, 1280)
    with open("models/filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
    return feature_list, filenames


feature_list, filenames = load_data()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="frs-header">
    <div style="font-size:2.3rem;line-height:1">👗</div>
    <div>
        <div class="frs-title">Fashion Recommendation System</div>
        <div class="frs-sub">Upload a fashion item · Discover similar styles instantly</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Card ───────────────────────────────────────────────────────────────────
st.markdown('<div class="frs-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "drop", type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

st.write("")

# ── Columns ────────────────────────────────────────────────────────────────
c_img, c_div, c1, c2, c3 = st.columns([2.0, 0.1, 2.1, 2.1, 2.1], gap="small")

with c_div:
    st.markdown('<div class="v-div"></div>', unsafe_allow_html=True)

with c_img:
    st.markdown('<div class="sec-label">📸 &nbsp;Your Style</div>', unsafe_allow_html=True)
    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, use_container_width=True)
    else:
        st.markdown("""
        <div class="ph-card">
            <span style="font-size:1.8rem">👗</span>
            <span>Drop your image here</span>
        </div>""", unsafe_allow_html=True)

result_cols = [c1, c2, c3]

if uploaded_file:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("_fkey") != file_key:
        st.session_state["_fkey"]    = file_key
        st.session_state["_indices"] = []
        st.session_state["_scores"]  = []
        st.session_state["_error"]   = ""

        with st.spinner("🔍 Finding your matches…"):
            feats = extract_features(pil_img)
            if feats is None:
                st.session_state["_error"] = "Feature extraction failed."
            else:
                idx, scores = recommend(feats, feature_list, top_k=3)
                st.session_state["_indices"] = idx
                st.session_state["_scores"]  = scores

    error   = st.session_state.get("_error",   "")
    indices = st.session_state.get("_indices", [])
    scores  = st.session_state.get("_scores",  [])

    if error:
        for col in result_cols:
            with col:
                st.error(error)
    elif not indices:
        for col in result_cols:
            with col:
                st.markdown('<div class="ph-card"><span>No results</span></div>',
                            unsafe_allow_html=True)
    else:
        for rank, (col, idx, score) in enumerate(zip(result_cols, indices, scores), 1):
            with col:
                st.markdown(f'<div class="sec-label">✨ &nbsp;Match #{rank}</div>',
                            unsafe_allow_html=True)
                try:
                    rec_img = Image.open(filenames[idx]).convert("RGB")
                    st.image(rec_img, use_container_width=True)
                except Exception:
                    st.markdown('<div class="ph-card"><span>Image not found</span></div>',
                                unsafe_allow_html=True)
                pct = round(float(score) * 100, 1)
                st.markdown(f"""
                <div class="match-badge">
                    <span class="match-rank">Rank #{rank}</span>
                    <span class="match-pct">{pct}% match</span>
                </div>""", unsafe_allow_html=True)
else:
    for rank, col in enumerate(result_cols, 1):
        with col:
            st.markdown(f'<div class="sec-label">✨ &nbsp;Match #{rank}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="ph-card">'
                '<span style="font-size:1.4rem">🔍</span>'
                '<span>Awaiting upload</span>'
                '</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)