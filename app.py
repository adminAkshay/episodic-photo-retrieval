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
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Episodic — Photo Retrieval",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"
SCORE_THRESHOLD = 0.60
TOP_K = 9

# ============================================================
# GOOGLE PHOTOS-STYLE CSS
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }

    /* Google Sans-like typography */
    html, body, [class*="css"] {
        font-family: 'Google Sans', 'Roboto', -apple-system, sans-serif;
    }

    /* Photo grid */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.15);
        transition: all 0.2s ease;
    }
    div[data-testid="stImage"] img:hover {
        box-shadow: 0 4px 12px rgba(60,64,67,0.25);
        transform: translateY(-2px);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 24px;
        border: 1px solid #dadce0;
        background-color: white;
        color: #5f6368;
        font-size: 13px;
        font-weight: 500;
        padding: 6px 16px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #f1f3f4;
        border-color: #dadce0;
        color: #202124;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* Search box */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #dadce0;
        padding: 12px 20px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.1);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e8eaed;
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
    return SentenceTransformer(EMBED_MODEL)


@st.cache_data(show_spinner="Loading photos from Pexels...")
def load_pexels_library(api_key: str, query: str, count: int = 40):
    """Fetch photos from Pexels and build the searchable library."""
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": min(count, 80), "orientation": "landscape"}
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
    except Exception as e:
        return [], None, str(e)

    if not photos:
        return [], None, "No photos found for that theme."

    library = []
    for p in photos:
        alt = p.get("alt") or ""
        library.append({
            "id": p["id"],
            "url": p["src"]["large"],
            "thumb_url": p["src"]["medium"],
            "title": alt[:60] if alt else f"Photo #{p['id']}",
            "photographer": p.get("photographer", "Unknown"),
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
    return library, vecs, None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


# ============================================================
# LLM: CUE EXTRACTION
# ============================================================
CUE_PROMPT = """You are an expert at translating how people describe photos into structured retrieval queries.

The user describes a photo using episodic memory (fuzzy, sensory, emotional). Extract structured cues.

Return ONLY valid JSON:
{
  "objects": ["physical objects mentioned or implied"],
  "scene": "setting (beach, kitchen, city street) or empty string",
  "people": ["who is in the photo"],
  "time_period": "when (summer, sunset, 2015, night) or empty string",
  "colors": ["dominant colors mentioned"],
  "mood": "emotional tone (warm, joyful, quiet) or empty string",
  "negation": ["things explicitly NOT in the photo"],
  "confidence": {"objects": 0.0, "scene": 0.0, "people": 0.0, "time_period": 0.0, "colors": 0.0},
  "clarifying_questions": ["up to 2 questions to reduce ambiguity"]
}

Rules:
- "yellow truck" → objects: ["yellow truck"], colors: ["yellow"]
- "me alone" → negation: ["no other people"], people: ["me"]
- confidence = your certainty, 0.0 to 1.0
- Return ONLY JSON, no prose.
"""


def extract_cues(user_text, history_context=""):
    client = get_groq()
    if client is None:
        return {"error": "no_api_key"}
    messages = [
        {"role": "system", "content": CUE_PROMPT},
        {"role": "user", "content": f"Context:\n{history_context}\n\nDescription:\n\"{user_text}\""},
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# RETRIEVAL
# ============================================================
def score_photo(photo, cues, semantic_vec, photo_vecs, idx):
    m = photo["metadata"]
    breakdown = {}
    score = 0.0

    sem = float(photo_vecs[idx] @ semantic_vec)
    score += 0.35 * max(0, sem)
    breakdown["semantic"] = f"{sem:.2f}"

    if cues.get("scene"):
        sq, sp = cues["scene"].lower(), m["scene"].lower()
        if sq in sp or sp in sq:
            score += 0.15
            breakdown["scene"] = "✓"
        else:
            breakdown["scene"] = "✗"

    if cues.get("objects"):
        obj_text = " ".join(m["objects"]).lower()
        matched = [o for o in cues["objects"] if any(w in obj_text for w in o.lower().split() if len(w) > 2)]
        if matched:
            score += 0.20 * (len(matched) / len(cues["objects"]))
            breakdown["objects"] = f"✓ {', '.join(matched)}"
        else:
            breakdown["objects"] = "✗"

    if cues.get("people"):
        ppl = " ".join(m["people"]).lower()
        matched = [p for p in cues["people"] if p.lower() in ppl]
        if matched:
            score += 0.15
            breakdown["people"] = f"✓ {', '.join(matched)}"
        else:
            breakdown["people"] = "✗"

    if cues.get("colors"):
        col = " ".join(m["colors"]).lower()
        matched = [c for c in cues["colors"] if c.lower() in col]
        if matched:
            score += 0.10 * (len(matched) / len(cues["colors"]))
            breakdown["colors"] = f"✓ {', '.join(matched)}"
        else:
            breakdown["colors"] = "✗"

    if cues.get("time_period"):
        if cues["time_period"].lower() in m["time_period"].lower():
            score += 0.05
            breakdown["time_period"] = "✓"
        else:
            breakdown["time_period"] = "✗"

    for neg in cues.get("negation", []):
        nl = neg.lower()
        combined = (m["scene"] + " " + " ".join(m["objects"]) + " " + " ".join(m["people"])).lower()
        if "no other people" in nl and len(m["people"]) > 1:
            score -= 0.20
            breakdown["negation"] = "✗ people present"
        elif any(w in combined for w in nl.split() if len(w) > 3):
            score -= 0.10
            breakdown["negation"] = "✗"

    return {"score": max(0.0, score), "breakdown": breakdown}


def retrieve(cues, query_vec, library, photo_vecs, top_k=TOP_K):
    results = []
    for i, photo in enumerate(library):
        s = score_photo(photo, cues, query_vec, photo_vecs, i)
        results.append({"photo": photo, **s})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ============================================================
# SESSION STATE
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
def render_results(results, cues):
    if not results:
        st.warning("No candidates found. Try describing the photo differently.")
        return

    top_score = results[0]["score"]
    st.markdown(f"**Found {len(results)} candidates** · best match **{top_score:.0%}**")

    if top_score < SCORE_THRESHOLD and cues.get("clarifying_questions"):
        with st.container(border=True):
            st.markdown("#### 🤔 Help me narrow this down")
            for q in cues["clarifying_questions"][:2]:
                st.markdown(f"- {q}")
            st.caption("Reply below — I'll use your answer to refine.")

    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            img = fetch_image(r["photo"]["thumb_url"])
            if img:
                st.image(img, use_container_width=True)
            st.caption(f"**{r['score']:.0%}** · {r['photo']['title']}")
            with st.expander("Why this matched"):
                st.json(r["breakdown"])
            if st.button(
                "More like this →",
                key=f"like_{i}_{r['photo']['id']}",
                use_container_width=True,
            ):
                st.session_state.pending_input = (
                    f"More like {r['photo']['title']}: {r['photo']['metadata']['description']}"
                )
                st.rerun()


def render_welcome():
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem 2rem 1rem;">
            <h1 style="font-size:3rem; margin-bottom:0.5rem;">🧠 Episodic</h1>
            <p style="font-size:1.15rem; color:#5f6368; max-width:700px; margin:0 auto;">
                Describe a photo the way you actually remember it — fuzzy, sensory, multi-cue.
                I'll break down your memory into structured cues and find the closest matches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💡 Try describing a photo like this:")
    samples = [
        "yellow truck on a beach, some summer years ago",
        "my dog in green grass, sunny day, looked really happy",
        "a wedding outdoors with white flowers and warm light",
        "city at night with bright lights, I was alone",
    ]
    cols = st.columns(2)
    for i, s in enumerate(samples):
        if cols[i % 2].button(s, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_input = s
            st.rerun()


# ============================================================
# MAIN
# ============================================================
def main():
    init_state()

    if get_groq() is None:
        st.error(
            "🔑 **Groq API key missing.**  \n"
            "Add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    # Check Pexels key
    try:
        pexels_key = st.secrets["PEXELS_API_KEY"]
    except Exception:
        pexels_key = os.getenv("PEXELS_API_KEY")

    if not pexels_key:
        st.error(
            "🔑 **Pexels API key missing.**  \n"
            "Add `PEXELS_API_KEY` in Streamlit Cloud → Settings → Secrets.  \n"
            "Get a free key at [pexels.com/api](https://www.pexels.com/api/)."
        )
        st.stop()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 🧠 Episodic")
        st.caption("Photo retrieval from fuzzy memory")
        st.divider()

        st.markdown("#### 📸 Photo Library")
        if st.session_state.theme is None:
            theme = st.text_input(
                "Choose a theme to load photos:",
                value="beach sunset",
                key="theme_input",
            )
            if st.button("Load Library", use_container_width=True):
                with st.spinner(f"Fetching '{theme}' photos from Pexels..."):
                    lib, vecs, err = load_pexels_library(pexels_key, theme)
                    if err:
                        st.error(f"Failed: {err}")
                    elif not lib:
                        st.warning("No photos found. Try another theme.")
                    else:
                        st.session_state.library = lib
                        st.session_state.photo_vecs = vecs
                        st.session_state.theme = theme
                        st.rerun()
        else:
            st.success(f"**{len(st.session_state.library)} photos** loaded")
            st.caption(f"Theme: *{st.session_state.theme}*")
            if st.button("🔄 Change theme", use_container_width=True):
                st.session_state.theme = None
                st.session_state.library = None
                st.session_state.photo_vecs = None
                st.session_state.history = []
                st.rerun()

        st.divider()
        st.markdown("#### ℹ️ About")
        st.caption(
            "An AI-native retrieval agent that fixes the "
            "**multi-cue** and **negation** failures in photo search."
        )
        st.caption("Powered by Groq + Llama 3.3 · Pexels · Local embeddings")

        if st.session_state.history:
            st.divider()
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    # ---------- MAIN ----------
    if st.session_state.theme is None:
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 1rem;">
                <h2>👈 Start by loading a photo library</h2>
                <p style="color:#5f6368;">Choose a theme in the sidebar to fetch photos from Pexels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if not st.session_state.history:
        render_welcome()

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("results"):
                render_results(turn["results"], turn.get("cues", {}))

    pending = st.session_state.pop("pending_input", None)
    prompt = st.chat_input("Describe the photo you're looking for...")
    if pending and not prompt:
        prompt = pending

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history_context = "\n".join(
            f"{t['role']}: {t['content']}" for t in st.session_state.history[-4:]
        )

        with st.chat_message("assistant"):
            with st.spinner("Parsing your memory and searching..."):
                cues = extract_cues(prompt, history_context)
                if "error" in cues:
                    st.error(f"Could not parse: {cues['error']}")
                    st.session_state.history.append(
                        {"role": "assistant", "content": "Sorry, I couldn't parse that."}
                    )
                    st.stop()

                embedder = get_embedder()
                query_text = prompt + " " + " ".join(cues.get("objects", [])) + " " + cues.get("scene", "")
                query_vec = embedder.encode([query_text], normalize_embeddings=True)[0]

                results = retrieve(
                    cues, query_vec,
                    st.session_state.library,
                    st.session_state.photo_vecs,
                )

                cue_parts = []
                for k in ["objects", "scene", "people", "time_period", "colors", "negation"]:
                    v = cues.get(k)
                    if v:
                        val = ", ".join(v) if isinstance(v, list) else v
                        cue_parts.append(f"*{k}*: {val}")
                response = f"I understood — {'; '.join(cue_parts) or '—'}.\n\nHere are my best candidates:"

                st.markdown(response)
                render_results(results, cues)

                st.session_state.history.append({
                    "role": "assistant",
                    "content": response,
                    "results": results,
                    "cues": cues,
                })

        st.rerun()


if __name__ == "__main__":
    main()
