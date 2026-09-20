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
    initial_sidebar_state="collapsed",
)

# ============================================================
# MOBILE GOOGLE PHOTOS UI
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, header, footer, [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    /* Dark surround */
    .stApp { background-color: #0f0f0f; }

    /* Phone frame */
    .main .block-container {
        max-width: 420px !important;
        padding: 0 !important;
        margin: 20px auto !important;
        background: #ffffff;
        border-radius: 36px;
        box-shadow:
            0 30px 80px rgba(0,0,0,0.7),
            0 0 0 10px #1c1c1e,
            0 0 0 12px #2c2c2e;
        overflow: hidden;
        min-height: 860px;
        position: relative;
    }

    /* App bar */
    .gp-appbar {
        background: #ffffff;
        padding: 18px 20px 6px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .gp-appbar-title {
        font-family: 'Google Sans', 'Roboto', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 400;
        color: #202124;
        letter-spacing: -0.3px;
    }
    .gp-appbar-logo { width: 24px; height: 24px; flex-shrink: 0; }

    /* Top search bar (gallery view) */
    .stTextInput { padding: 12px 16px 4px 16px; }
    .stTextInput > div > div > input {
        background: #f1f3f4 !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 12px 18px 12px 44px !important;
        font-size: 14px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif !important;
        color: #202124 !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='%235f6368'%3E%3Cpath d='M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 16px center !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #5f6368 !important;
    }

    /* Section headers */
    .gp-section {
        padding: 10px 16px 6px 16px;
        font-family: 'Google Sans', 'Roboto', sans-serif;
    }
    .gp-section h3 {
        font-size: 14px;
        font-weight: 500;
        color: #202124;
        margin: 0;
    }
    .gp-section p {
        font-size: 11px;
        color: #5f6368;
        margin: 2px 0 0 0;
    }

    /* Grid images */
    [data-testid="stImage"] img {
        border-radius: 6px !important;
        transition: transform 0.15s ease;
    }
    [data-testid="stImage"] img:hover { transform: scale(1.03); }

    /* Small "More like this" button */
    .stButton > button {
        background: transparent;
        border: none;
        color: #1a73e8;
        font-size: 10px;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-weight: 500;
        padding: 2px 0;
        width: 100%;
        text-align: left;
    }
    .stButton > button:hover {
        background: transparent;
        text-decoration: underline;
    }

    /* Secondary buttons (Back, sample prompts) */
    .stButton > button[kind="secondary"] {
        background: #f1f3f4;
        border: none;
        border-radius: 20px;
        padding: 8px 14px;
        color: #202124;
        font-size: 12px;
        font-weight: 500;
        width: auto;
        text-align: center;
        margin: 4px 16px;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #e8eaed;
        text-decoration: none;
    }

    /* ==================================================== */
    /* CHAT INPUT — BOTTOM BAR — TEXT IN BLACK (FIX)        */
    /* ==================================================== */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border-top: 1px solid #f0f0f0;
        padding: 8px 16px 12px 16px;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: #f1f3f4 !important;
        border: none !important;
        border-radius: 24px !important;
        font-size: 13px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif !important;
        color: #202124 !important;          /* ← BLACK TEXT */
        -webkit-text-fill-color: #202124 !important;  /* ← Fix for some browsers */
        caret-color: #1a73e8 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #5f6368 !important;
        -webkit-text-fill-color: #5f6368 !important;
        opacity: 1 !important;
    }

    /* ==================================================== */
    /* CHAT MESSAGES — CONVERSATION BUBBLES                */
    /* ==================================================== */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 12px 16px !important;
        margin: 6px 12px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-size: 13px;
        color: #202124;
    }

    /* User bubble — light blue */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #e8f0fe;
    }

    /* Assistant bubble — light grey */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #f8f9fa;
    }

    [data-testid="stChatMessage"] p {
        font-size: 13px;
        color: #202124 !important;
        margin: 0;
        line-height: 1.45;
    }

    /* Metadata under photos */
    .photo-meta {
        font-size: 10px;
        color: #5f6368;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        margin: 4px 0 2px 0;
        line-height: 1.3;
    }

    /* Clarifying questions */
    .clarify-box {
        background: #e8f0fe;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: 'Google Sans', sans-serif;
    }
    .clarify-title {
        color: #1967d2;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .clarify-q {
        color: #1967d2;
        font-size: 11px;
        margin: 3px 0;
    }

    /* Expander */
    details { border: none !important; }
    details summary {
        font-size: 10px !important;
        color: #5f6368 !important;
        font-family: 'Google Sans', sans-serif;
        padding: 2px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DEFAULT LIBRARY
# ============================================================
DEFAULT_QUERIES = [
    "family outdoor",
    "beach sunset",
    "dog pet",
    "birthday celebration",
    "city night lights",
    "mountain landscape",
]
PHOTOS_PER_QUERY = 8


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


@st.cache_data(show_spinner="Loading your photo library...")
def load_default_library(pexels_key):
    library = []
    seen_ids = set()

    for query in DEFAULT_QUERIES:
        headers = {"Authorization": pexels_key}
        params = {"query": query, "per_page": PHOTOS_PER_QUERY, "orientation": "landscape"}
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                headers=headers, params=params, timeout=15,
            )
            r.raise_for_status()
            photos = r.json().get("photos", [])
        except Exception:
            continue

        for p in photos:
            if p["id"] in seen_ids:
                continue
            seen_ids.add(p["id"])
            alt = (p.get("alt") or "").strip()
            library.append({
                "id": p["id"],
                "url": p["src"]["large"],
                "thumb_url": p["src"]["medium"],
                "title": alt[:50] if alt else f"Photo #{p['id']}",
                "photographer": p.get("photographer", ""),
                "metadata": {
                    "scene": query,
                    "objects": [alt] if alt else [],
                    "people": [],
                    "time_period": "daytime",
                    "colors": [],
                    "mood": "",
                    "description": f"{query}. {alt}",
                },
            })

    if not library:
        return [], None, "Could not fetch any photos."

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
        "library": None,
        "photo_vecs": None,
        "messages": [],        # conversation history
        "pending_input": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# UI HELPERS
# ============================================================
def render_appbar():
    st.markdown("""
    <div class="gp-appbar">
        <svg class="gp-appbar-logo" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path fill="#4285F4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c5.52 0 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93z"/>
            <path fill="#EA4335" d="M17.9 17.39c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
        </svg>
        <span class="gp-appbar-title">Photos</span>
    </div>
    """, unsafe_allow_html=True)


def render_grid(photos, show_scores=False, unique_prefix="grid"):
    if not photos:
        st.markdown(
            '<div class="gp-section"><p>No photos found.</p></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div style="padding: 0 12px;">', unsafe_allow_html=True)
    cols = st.columns(3, gap="small")
    for i, r in enumerate(photos):
        photo = r["photo"] if "photo" in r else r
        score = r.get("score") if show_scores else None
        breakdown = r.get("breakdown") if show_scores else None

        with cols[i % 3]:
            img = fetch_image(photo["thumb_url"])
            if img:
                st.image(img, use_container_width=True)

            if show_scores and score is not None:
                st.markdown(
                    f'<div class="photo-meta" style="color:#1a73e8; font-weight:500;">'
                    f'{score:.0%} match</div>',
                    unsafe_allow_html=True,
                )

            if show_scores and breakdown:
                with st.expander("Why"):
                    st.json(breakdown)
                if st.button(
                    "More like this",
                    key=f"like_{unique_prefix}_{i}_{photo['id']}",
                ):
                    st.session_state.pending_input = (
                        f"More like {photo['title']}: {photo['metadata']['description']}"
                    )
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

    # Auto-load library
    if st.session_state.library is None:
        lib, vecs, err = load_default_library(pexels_key)
        if err or not lib:
            st.error(f"⚠️ Could not load photos: {err}")
            st.stop()
        st.session_state.library = lib
        st.session_state.photo_vecs = vecs

    render_appbar()

    # ============================================================
    # GALLERY VIEW (no messages yet)
    # ============================================================
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="gp-section">
            <h3>Recent</h3>
            <p>{len(st.session_state.library)} photos · tap search or describe a memory below</p>
        </div>
        """, unsafe_allow_html=True)

        render_grid(st.session_state.library, show_scores=False, unique_prefix="home")

        # Suggested queries
        st.markdown('<div class="gp-section"><h3>Try asking</h3></div>', unsafe_allow_html=True)
        samples = [
            "a sunny beach at sunset",
            "my dog in green grass",
            "city at night, I was alone",
            "warm light on a face",
        ]
        for i, s in enumerate(samples):
            if st.button(s, key=f"sample_{i}", type="secondary", use_container_width=True):
                st.session_state.pending_input = s
                st.rerun()

    # ============================================================
    # CONVERSATION VIEW
    # ============================================================
    else:
        # Back button
        if st.button("← Back to gallery", key="back_btn", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Render conversation
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Clarifying questions
                if msg.get("clarifying") and msg["clarifying"].get("clarifying_questions"):
                    st.markdown('<div class="clarify-box">'
                                '<div class="clarify-title">💡 Help me narrow this down</div>',
                                unsafe_allow_html=True)
                    for q in msg["clarifying"]["clarifying_questions"][:2]:
                        st.markdown(f'<div class="clarify-q">• {q}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Results grid inside the assistant bubble
                if msg.get("results"):
                    render_grid(
                        msg["results"],
                        show_scores=True,
                        unique_prefix=f"turn_{idx}",
                    )

    # ============================================================
    # INPUT — pick up pending or new input
    # ============================================================
    pending = st.session_state.pop("pending_input", None)
    chat_prompt = st.chat_input("Describe a photo you're looking for...")

    user_query = pending or chat_prompt

    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Process
        with st.spinner("Parsing your memory..."):
            history_ctx = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-4:]
            )
            cues = extract_cues(user_query, history_ctx)

            if "error" in cues:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Could not parse that: {cues['error']}",
                })
                st.rerun()

            embedder = get_embedder()
            qtext = (
                user_query
                + " "
                + " ".join(cues.get("objects", []))
                + " "
                + cues.get("scene", "")
            )
            qvec = embedder.encode([qtext], normalize_embeddings=True)[0]

            results = retrieve(
                cues, qvec,
                st.session_state.library,
                st.session_state.photo_vecs,
            )

            # Build friendly response text
            parts = []
            for k in ["objects", "scene", "people", "time_period", "colors", "negation"]:
                v = cues.get(k)
                if v:
                    val = ", ".join(v) if isinstance(v, list) else v
                    parts.append(f"**{k}**: {val}")

            top_score = results[0]["score"] if results else 0
            if top_score > 0.6:
                response = f"Found {len(results)} matches. Here's what I understood — {'; '.join(parts) or '—'}."
            else:
                response = f"I found {len(results)} candidates, though none are a strong match. Here's what I understood — {'; '.join(parts) or '—'}."

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "results": results,
                "clarifying": cues if top_score < 0.6 else None,
            })

        st.rerun()


if __name__ == "__main__":
    main()
