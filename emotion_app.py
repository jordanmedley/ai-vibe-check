import base64
import colorsys
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from inference_sdk import InferenceHTTPClient

# =========================
# CONFIG
# =========================
ROBOFLOW_API_KEY = "jdS0hFcHsW2SurIhmkaR"
API_URL = "https://serverless.roboflow.com"
WORKSPACE_NAME = "jordans-vibe-check"
WORKFLOW_ID = "detect-and-classify"

st.set_page_config(
    page_title="AI Vibe Check",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# GLOBAL STYLES (enterprise look)
# =========================
st.markdown(
    """
    <style>
      :root {
        --bg: #0b0f14;
        --panel: #111821;
        --panel-2: #0e141b;
        --text: #e6ebf0;
        --muted: #96a0aa;
        --brand: #6E9BFF; /* accent */
        --ok: #2ecc71;
        --warn: #f1c40f;
        --err: #e74c3c;
        --radius: 14px;
      }
      .stApp { background: radial-gradient(1200px 900px at 20% -10%, #121a24, var(--bg)); color: var(--text); }
      header, footer { visibility: hidden; height: 0; }
      .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }
      .app-title {
        display:flex; align-items:center; gap:.75rem; margin-bottom:.75rem;
        font-weight:700; font-size:1.55rem; letter-spacing:.2px;
      }
      .app-sub { color: var(--muted); margin-bottom: 1.25rem; }
      .card {
        background: linear-gradient(180deg, var(--panel), var(--panel-2));
        border: 1px solid rgba(255,255,255,.04);
        border-radius: var(--radius);
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
      }
      .card.tight { padding: .9rem 1rem; }
      .pill {
        display:inline-flex; align-items:center; gap:.45rem; padding:.2rem .55rem;
        border-radius: 999px; background: rgba(255,255,255,.06); color: var(--muted);
        border: 1px solid rgba(255,255,255,.06); font-size:.85rem;
      }
      .metric {
        display:flex; flex-direction:column; gap:.25rem;
        padding:.6rem .8rem; background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.06); border-radius: 12px;
      }
      .metric .k { font-size:1.05rem; color: var(--muted); }
      .metric .v { font-weight:700; font-size:1.2rem; }
      .swatch {
        width: 46px; height: 46px; border-radius: 10px; border: 1px solid rgba(255,255,255,.16);
      }
      .muted { color: var(--muted); }
      .center { display:flex; justify-content:center; align-items:center; }
      .hero-img img { border-radius: 16px; border: 1px solid rgba(255,255,255,.08); }
      /* remove default uploader border */
      .stFileUploader > div > div { background: transparent; }
      .rightline { border-left: 1px dashed rgba(255,255,255,.08); }
      .brand { color: var(--brand); font-weight: 600; }
      /* hide hamburger + streamlit footer menu */
      #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CLIENT (cached)
# =========================
@st.cache_resource(show_spinner=False)
def get_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url=API_URL, api_key=ROBOFLOW_API_KEY)

# =========================
# UTILITIES
# =========================
def rgb_tone_text(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h *= 360
    if h < 30 or h > 330:
        return "Warm & energetic"
    if 30 <= h < 90:
        return "Fresh & friendly"
    if 90 <= h < 180:
        return "Calm & balanced"
    if 180 <= h < 270:
        return "Cool & composed"
    return "Moody & intriguing"

def dominant_rgb(img: Image.Image) -> Tuple[int, int, int]:
    thumb = img.copy().resize((120, 120))
    arr = np.array(thumb).reshape((-1, 3))
    r, g, b = np.mean(arr, axis=0)
    return int(r), int(g), int(b)

def summarize_vibe(emotion: str, tone: str) -> str:
    t = tone.lower()
    messages = {
        "happy": f"Reads {t} and approachable. This is a strong first-impression photo.",
        "sad": f"Leans {t} and introspective. Consider brighter light or eye contact for dating apps.",
        "angry": f"High energy with a {t} feel. A smile shot could broaden appeal.",
        "surprised": f"Curious and {t}. Natural, candid energy works here.",
        "neutral": f"Clean, composed, and {t}. Subtle smile could lift warmth.",
        "fear": f"{t} with reserved energy. Softer lighting may help.",
        "disgust": f"{t} but distant. Try a clean background and smile.",
    }
    return messages.get(emotion.lower(), f"Unique vibe with a {t} undertone.")

def extract_predictions(result: dict | list) -> List[Dict]:
    """Robustly find the predictions array regardless of SDK response shape."""
    if isinstance(result, list) and result:
        # Workflow payload array
        node = result[0]
        dp = node.get("detection_predictions") or node.get("results", [{}])[0]
        return (dp or {}).get("predictions") or []
    if isinstance(result, dict):
        return (
            result.get("detection_predictions", {}).get("predictions")
            or (result.get("results", [{}])[0] or {}).get("predictions")
            or []
        )
    return []

def run_workflow_on_image(img_bytes: io.BytesIO) -> dict | list:
    client = get_client()
    img_bytes.seek(0)
    b64 = base64.b64encode(img_bytes.read()).decode("utf-8")
    return client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": b64},  # matches your working workflow output
    )

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="app-title">💘 AI Vibe Check</div>
    <div class="app-sub">Professional-grade photo vibe analysis — emotion + color psychology.</div>
    """,
    unsafe_allow_html=True,
)

# =========================
# LAYOUT
# =========================
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.markdown("#### Upload")
    up_card = st.container(border=False)
    with up_card:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a selfie or profile photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.markdown('<div class="hero-img">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Drop a photo to begin. Aim for a clear, front-facing face in good light.")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("#### Result")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if not uploaded:
        st.markdown(
            """
            <div class="muted">No image yet. Your emotion, color tone, confidence, and recommendations will appear here.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing your vibe…"):
            try:
                # run workflow
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                result = run_workflow_on_image(img_bytes)

                preds = extract_predictions(result)
                if not preds:
                    st.warning("No face detected. Try a closer crop or stronger lighting.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    top = max(preds, key=lambda p: p.get("confidence", 0.0))
                    emotion = top.get("class", "Unknown")
                    conf = float(top.get("confidence", 0.0))

                    r, g, b = dominant_rgb(img)
                    tone = rgb_tone_text(r, g, b)

                    # headline
                    st.markdown(
                        f"""
                        <div class="pill">Workflow: <span class="brand">{WORKFLOW_ID}</span></div>
                        <h3 style="margin:.5rem 0 0 0;">{emotion.capitalize()}</h3>
                        <div class="muted" style="margin-top:.2rem;">Detected primary emotion</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.1%}")

                    st.markdown("---")

                    # metrics row
                    m1, m2, m3, m4 = st.columns([1.1, 1.1, 1.1, 1.1])
                    with m1:
                        st.markdown('<div class="metric"><div class="k">Tone</div><div class="v">'
                                    f'{tone}</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric"><div class="k">RGB</div>'
                                    f'<div class="v">{r}, {g}, {b}</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric"><div class="k">Faces</div>'
                                    f'<div class="v">{len(preds)}</div></div>', unsafe_allow_html=True)
                    with m4:
                        st.markdown(
                            f'''
                            <div class="metric"><div class="k">Palette</div>
                            <div class="v" style="display:flex;align-items:center;gap:.6rem">
                                <div class="swatch" style="background: rgb({r},{g},{b});"></div>
                            </div></div>
                            ''',
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

                    # narrative recommendation
                    st.markdown("#### Recommendation")
                    st.success(summarize_vibe(emotion, tone))

                    # raw (collapsible)
                    with st.expander("Technical output (JSON)"):
                        st.json(result)

                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Something went wrong while analyzing the image: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown(
    """
    <div style="margin-top:1.5rem; color: var(--muted); font-size:.9rem;">
      Powered by <span class="brand">Roboflow</span> • Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

