import json
import os
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from groq import Groq
from PIL import Image
from sentence_transformers import SentenceTransformer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Episodic Photos",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# MOBILE GOOGLE PHOTOS UI
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, header, footer, [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    /* Body background — dark grey like phone screenshot context */
    .stApp {
        background-color: #202124;
    }

    /* Phone frame container */
    .main .block-container {
        max-width: 430px !important;
        padding: 0 !important;
        margin: 20px auto !important;
        background: #ffffff;
        border-radius: 32px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 8px #1a1a1a, 0 0 0 10px #2a2a2a;
        overflow: hidden;
        min-height: 850px;
    }

    /* Google Photos top app bar */
    .gp-appbar {
        background: #ffffff;
        padding: 16px 20px 12px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    .gp-appbar-title {
        font-family: 'Google Sans', 'Roboto', -apple-system, sans-serif;
        font-size: 20px;
        font-weight: 400;
        color: #202124;
        letter-spacing: -0.2px;
    }
    .gp-appbar-logo {
        width: 26px;
        height: 26px;
        flex-shrink: 0;
    }

    /* Google Photos search bar */
    .stTextInput > div > div > input {
        background: #f1f3f4 !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 12px 20px 12px 48px !important;
        font-size: 15px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif !important;
        color: #202124 !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='%235f6368'%3E%3Cpath d='M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 18px center !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #5f6368 !important;
    }
    .stTextInput {
        padding: 12px 16px 0 16px;
    }

    /* Section headers */
    .gp-section {
        padding: 16px 20px 8px 20px;
        font-family: 'Google Sans', 'Roboto', sans-serif;
    }
    .gp-section h3 {
        font-size: 15px;
        font-weight: 500;
        color: #202124;
        margin: 0;
    }
    .gp-section p {
        font-size: 12px;
        color: #5f6368;
        margin: 2px 0 0 0;
    }

    /* Photo grid captions */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        transition: transform 0.15s ease;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }

    /* Small buttons ("More like this") */
    .stButton > button {
        background: transparent;
        border: none;
        color: #1a73e8;
        font-size: 11px;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-weight: 500;
        padding: 2px 0;
        width: 100%;
        text-align: left;
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        background: transparent;
        color: #174ea6;
        text-decoration: underline;
    }

    /* Sample prompt buttons on welcome */
    .stButton > button[kind="secondary"] {
        background: #f8f9fa;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 12px 14px;
        color: #202124;
        font-size: 13px;
        text-align: left;
        height: auto;
        white-space: normal;
        line-height: 1.4;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #f1f3f4;
        border-color: #dadce0;
        text-decoration: none;
    }

    /* Chat messages — Google Photos card style */
    [data-testid="stChatMessage"] {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 12px 16px !important;
        margin: 8px 16px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-size: 13px;
    }
    [data-testid="stChatMessage"] p {
        font-size: 13px;
        color: #202124;
        margin: 0;
    }

    /* Expander ("Why this matched") */
    details {
        border: none !important;
    }
    details summary {
        font-size: 11px !important;
        color: #5f6368 !important;
        font-family: 'Google Sans', sans-serif;
        padding: 2px 0 !important;
    }

    /* Chat input bar — pinned bottom like phone */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border-top: 1px solid #f0f0f0;
        padding: 12px 16px;
    }
    [data-testid="stChatInput"] textarea {
        background: #f1f3f4 !important;
        border: none !important;
        border-radius: 24px !important;
        font-size: 14px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif !important;
        padding: 12px 18px !important;
    }

    /* Welcome hero */
    .gp-hero {
        text-align: center;
        padding: 30px 24px 20px 24px;
        font-family: 'Google Sans', 'Roboto', sans-serif;
    }
    .gp-hero h1 {
        font-size: 24px;
        font-weight: 400;
        color: #202124;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
    }
    .gp-hero p {
        font-size: 13px;
        color: #5f6368;
        line-height: 1.5;
        margin: 0;
    }

    /* Sidebar — keeps the phone frame clean */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8eaed;
    }
    section[data-testid="stSidebar"] .block-container {
        box-shadow: none !important;
        border-radius: 0 !important;
        margin: 0 !important;
    }

    /* Scrollbar hidden for phone feel */
    .main .block-container::-webkit-scrollbar {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# RESOURCES
# ============================================================
@st.cache_resource
def get_groq():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


@st.cache_data(show_spinner="Loading photos from Pexels...")
def load_pexels_library(api_key, query, count=25):
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": min(count, 80), "orientation": "landscape"}
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers, params=params, timeout=15,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
    except Exception as e:
        return [], None, str(e)

    if not photos:
        return [], None, "No photos found for that theme."

    library = []
    for p in photos:
        alt = (p.get("alt") or "").strip()
        library.append({
            "id": p["id"],
            "url": p["src"]["large"],
            "thumb_url": p["src"]["medium"],
            "title": alt[:60] if alt else f"Photo #{p['id']}",
            "photographer": p.get("photographer", ""),
            "metadata": {
                "scene": query,
                "objects": [alt] if alt else [],
                "people": [],
                "time_period": "daytime",
                "colors": [],
                "mood": "",
                "description": f"Stock photo related to '{query}'. {alt}",
            },
        })

    embedder = get_embedder()
    texts = [p["metadata"]["description"] for p in library]
    vecs = embedder.encode(texts, normalize_embeddings=True)
    return library, np.array(vecs), None


# ============================================================
# LLM: CUE EXTRACTION
# ============================================================
CUE_PROMPT = """Extract structured cues from a user's fuzzy description of a photo.

Return ONLY valid JSON:
{
  "objects": ["physical objects mentioned"],
  "scene": "setting or empty string",
  "people": ["who is in the photo"],
  "time_period": "when or empty string",
  "colors": ["colors mentioned"],
  "mood": "emotional tone or empty string",
  "negation": ["things explicitly NOT in the photo"],
  "confidence": {"objects":0.0,"scene":0.0,"people":0.0,"time_period":0.0,"colors":0.0},
  "clarifying_questions": ["up to 2 questions"]
}

Rules:
- "yellow truck" → objects:["yellow truck"], colors:["yellow"]
- "me alone" → negation:["no other people"], people:["me"]
- Return ONLY JSON."""


def extract_cues(text, history=""):
    client = get_groq()
    if not client:
        return {"error": "no_api_key"}
    messages = [
        {"role": "system", "content": CUE_PROMPT},
        {"role": "user", "content": f"Context:\n{history}\n\nDescription: \"{text}\""},
    ]
    try:
        r = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# RETRIEVAL
# ============================================================
def score_photo(photo, cues, qvec, pvecs, idx):
    m = photo["metadata"]
    bd = {}
    score = 0.0

    sem = float(pvecs[idx] @ qvec)
    score += 0.35 * max(0, sem)
    bd["semantic"] = f"{sem:.2f}"

    if cues.get("scene"):
        sq, sp = cues["scene"].lower(), m["scene"].lower()
        if sq in sp or sp in sq:
            score += 0.15
            bd["scene"] = "✓"
        else:
            bd["scene"] = "✗"

    if cues.get("objects"):
        ot = " ".join(m["objects"]).lower()
        matched = [o for o in cues["objects"] if any(w in ot for w in o.lower().split() if len(w) > 2)]
        if matched:
            score += 0.20 * (len(matched) / len(cues["objects"]))
            bd["objects"] = f"✓ {', '.join(matched)}"
        else:
            bd["objects"] = "✗"

    if cues.get("people"):
        ppl = " ".join(m["people"]).lower()
        matched = [p for p in cues["people"] if p.lower() in ppl]
        if matched:
            score += 0.15
            bd["people"] = f"✓ {', '.join(matched)}"
        else:
            bd["people"] = "✗"

    if cues.get("colors"):
        col = " ".join(m["colors"]).lower()
        matched = [c for c in cues["colors"] if c.lower() in col]
        if matched:
            score += 0.10 * (len(matched) / len(cues["colors"]))
            bd["colors"] = f"✓ {', '.join(matched)}"
        else:
            bd["colors"] = "✗"

    if cues.get("time_period"):
        if cues["time_period"].lower() in m["time_period"].lower():
            score += 0.05
            bd["time_period"] = "✓"
        else:
            bd["time_period"] = "✗"

    for neg in cues.get("negation", []):
        nl = neg.lower()
        combined = (m["scene"] + " " + " ".join(m["objects"]) + " " + " ".join(m["people"])).lower()
        if "no other people" in nl and len(m["people"]) > 1:
            score -= 0.20
            bd["negation"] = "✗ people present"
        elif any(w in combined for w in nl.split() if len(w) > 3):
            score -= 0.10
            bd["negation"] = "✗"

    return {"score": max(0.0, score), "breakdown": bd}


def retrieve(cues, qvec, library, pvecs, k=6):
    out = []
    for i, p in enumerate(library):
        s = score_photo(p, cues, qvec, pvecs, i)
        out.append({"photo": p, **s})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:k]


# ============================================================
# SESSION
# ============================================================
def init_state():
    defaults = {
        "history": [],
        "library": None,
        "photo_vecs": None,
        "theme": None,
        "pending_input": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# UI COMPONENTS
# ============================================================
def render_appbar():
    st.markdown("""
    <div class="gp-appbar">
        <svg class="gp-appbar-logo" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" fill="none"/>
            <path fill="#4285F4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c5.52 0 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93z"/>
            <path fill="#EA4335" d="M17.9 17.39c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
        </svg>
        <span class="gp-appbar-title">Episodic Photos</span>
    </div>
    """, unsafe_allow_html=True)


def render_results(results, cues, ctx="default"):
    if not results:
        st.markdown(
            '<div class="gp-section"><p>No matches found. Try describing it differently.</p></div>',
            unsafe_allow_html=True,
        )
        return

    top = results[0]["score"]
    st.markdown(f"""
    <div class="gp-section">
        <h3>Best matches</h3>
        <p>{len(results)} candidates · top match {top:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    if top < 0.60 and cues.get("clarifying_questions"):
        st.markdown("""
        <div style="background:#e8f0fe; border-radius:12px; padding:12px 16px;
                    margin:0 16px 12px 16px; font-family:'Google Sans',sans-serif;">
            <div style="color:#1967d2; font-size:13px; font-weight:500; margin-bottom:6px;">
                💡 Help me narrow this down
            </div>
        """, unsafe_allow_html=True)
        for q in cues["clarifying_questions"][:2]:
            st.markdown(
                f'<div style="color:#1967d2; font-size:12px; margin:4px 0;">• {q}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Photo grid — 2 columns for phone feel
    st.markdown('<div style="padding: 0 16px;">', unsafe_allow_html=True)
    cols = st.columns(2, gap="small")
    for i, r in enumerate(results):
        with cols[i % 2]:
            img = fetch_image(r["photo"]["thumb_url"])
            if img:
                st.image(img, use_container_width=True)
            st.markdown(
                f'<div style="font-size:11px; color:#5f6368; font-family:\'Google Sans\',sans-serif;'
                f' margin:4px 0 2px 0;">{r["score"]:.0%} · {r["photo"]["title"][:40]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Why this matched"):
                st.json(r["breakdown"])
            if st.button(
                "More like this →",
                key=f"like_{ctx}_{i}_{r['photo']['id']}",
            ):
                st.session_state.pending_input = (
                    f"More like {r['photo']['title']}: {r['photo']['metadata']['description']}"
                )
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div class="gp-hero">
        <h1>Search your memories</h1>
        <p>Describe a photo the way you remember it — fuzzy, sensory, multi-cue.
        I'll break down your memory into cues and find the closest matches.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gp-section"><h3>Try asking</h3></div>', unsafe_allow_html=True)

    samples = [
        "yellow truck on a beach, some summer years ago",
        "my dog in green grass, sunny day, looked happy",
        "a wedding outdoors with white flowers and warm light",
        "city at night with bright lights, I was alone",
    ]
    st.markdown('<div style="padding: 0 16px;">', unsafe_allow_html=True)
    for i, s in enumerate(samples):
        if st.button(s, key=f"sample_{i}", use_container_width=True, type="secondary"):
            st.session_state.pending_input = s
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
def main():
    init_state()

    if get_groq() is None:
        st.error("🔑 **Groq API key missing.** Add `GROQ_API_KEY` in Streamlit secrets.")
        st.stop()

    try:
        pexels_key = st.secrets["PEXELS_API_KEY"]
    except Exception:
        pexels_key = os.getenv("PEXELS_API_KEY")

    if not pexels_key:
        st.error("🔑 **Pexels API key missing.** Add `PEXELS_API_KEY` in Streamlit secrets.")
        st.stop()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 📸 Photo Library")
        if st.session_state.theme is None:
            theme = st.text_input("Theme to load:", value="beach sunset")
            if st.button("Load Library", use_container_width=True, type="primary"):
                with st.spinner(f"Fetching '{theme}'..."):
                    lib, vecs, err = load_pexels_library(pexels_key, theme)
                    if err:
                        st.error(err)
                    elif lib:
                        st.session_state.library = lib
                        st.session_state.photo_vecs = vecs
                        st.session_state.theme = theme
                        st.rerun()
        else:
            st.success(f"{len(st.session_state.library)} photos loaded")
            st.caption(f"Theme: *{st.session_state.theme}*")
            if st.button("🔄 Change theme", use_container_width=True):
                st.session_state.theme = None
                st.session_state.library = None
                st.session_state.photo_vecs = None
                st.session_state.history = []
                st.rerun()

        st.divider()
        st.caption("🧠 Groq + Llama 3.3 70B · Pexels · Local embeddings")

        if st.session_state.history:
            st.divider()
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    # ---------- APP BAR ----------
    render_appbar()

    # ---------- MAIN CONTENT ----------
    if st.session_state.theme is None:
        st.markdown("""
        <div class="gp-hero" style="padding-top:120px;">
            <h1>👈 Load a photo library</h1>
            <p>Choose a theme in the sidebar to fetch photos from Pexels.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if not st.session_state.history:
        render_welcome()
    else:
        for idx, turn in enumerate(st.session_state.history):
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("results"):
                    render_results(turn["results"], turn.get("cues", {}), ctx=f"hist_{idx}")

    # ---------- CHAT INPUT ----------
    pending = st.session_state.pop("pending_input", None)
    prompt = st.chat_input("Describe a photo you're looking for...")
    if pending and not prompt:
        prompt = pending

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history_ctx = "\n".join(
            f"{t['role']}: {t['content']}" for t in st.session_state.history[-4:]
        )

        with st.chat_message("assistant"):
            with st.spinner("Parsing your memory..."):
                cues = extract_cues(prompt, history_ctx)
                if "error" in cues:
                    st.error(f"Could not parse: {cues['error']}")
                    st.stop()

                embedder = get_embedder()
                qtext = prompt + " " + " ".join(cues.get("objects", [])) + " " + cues.get("scene", "")
                qvec = embedder.encode([qtext], normalize_embeddings=True)[0]

                results = retrieve(
                    cues, qvec,
                    st.session_state.library,
                    st.session_state.photo_vecs,
                )

                parts = []
                for k in ["objects", "scene", "people", "time_period", "colors", "negation"]:
                    v = cues.get(k)
                    if v:
                        val = ", ".join(v) if isinstance(v, list) else v
                        parts.append(f"*{k}*: {val}")

                response = f"Understood — {'; '.join(parts) or '—'}.\n\nHere are my best candidates:"
                st.markdown(response)
                render_results(
                    results, cues,
                    ctx=f"live_{len(st.session_state.history)}"
                )

                st.session_state.history.append({
                    "role": "assistant",
                    "content": response,
                    "results": results,
                    "cues": cues,
                })

        st.rerun()


if __name__ == "__main__":
    main()
