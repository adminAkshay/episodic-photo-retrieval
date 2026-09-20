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
    #MainMenu, header, footer, [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    .stApp { background-color: #0f0f0f; }

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
    }

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

    [data-testid="stImage"] img {
        border-radius: 6px !important;
        transition: transform 0.15s ease;
    }
    [data-testid="stImage"] img:hover { transform: scale(1.03); }

    /* Small buttons — "More like this" */
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

    /* Secondary buttons — samples and back */
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

    /* Chat input — BLACK TEXT */
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
        color: #202124 !important;
        -webkit-text-fill-color: #202124 !important;
        caret-color: #1a73e8 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #5f6368 !important;
        -webkit-text-fill-color: #5f6368 !important;
        opacity: 1 !important;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 12px 16px !important;
        margin: 6px 12px !important;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        font-size: 13px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #e8f0fe;
    }
    [data-testid="stChatMessage"] p {
        font-size: 13px;
        color: #202124 !important;
        margin: 0;
        line-height: 1.45;
    }

    .photo-meta {
        font-size: 10px;
        color: #5f6368;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        margin: 4px 0 2px 0;
        line-height: 1.3;
    }

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
# CONFIG
# ============================================================
GALLERY_QUERIES = [
    "family outdoor",
    "beach sunset",
    "dog pet",
    "birthday celebration",
    "city night lights",
    "mountain landscape",
]
GALLERY_PER_QUERY = 8
LIVE_RESULTS = 6


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


# ============================================================
# PEXELS HELPERS
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def search_pexels(api_key, query, count=20):
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": count, "orientation": "landscape"}
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers, params=params, timeout=15,
        )
        r.raise_for_status()
        return r.json().get("photos", [])
    except Exception:
        return []


COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "gray", "grey", "golden", "silver",
    "dark", "bright", "colorful",
]
PEOPLE_WORDS = [
    "man", "woman", "person", "people", "child", "kid", "boy", "girl",
    "family", "couple", "baby", "crowd", "friends", "group",
]
OBJECT_WORDS = [
    "truck", "car", "vehicle", "dog", "cat", "pet", "puppy", "kitten",
    "tree", "flower", "mountain", "ocean", "beach", "sea", "sun",
    "sky", "food", "cake", "book", "house", "building", "street",
    "city", "bike", "boat", "water", "road", "path", "plant",
    "night", "sunset", "sunrise", "light",
]


def parse_alt(alt, query):
    combined = (alt + " " + query).lower()
    colors = [c for c in COLOR_WORDS if c in combined]
    people = [p for p in PEOPLE_WORDS if p in combined]
    objects = [o for o in OBJECT_WORDS if o in combined]
    for word in query.lower().split():
        if len(word) > 3 and word not in colors + people + objects:
            objects.append(word)
    return objects, colors, people


def normalize_scores(results):
    """Map raw scores to 85-95% for the top match, scale the rest."""
    if not results:
        return results
    top_raw = results[0]["score"]
    if top_raw <= 0:
        return results

    # Confidence scales with raw score — 0.15 → 85%, 0.7+ → 95%
    confidence = min(1.0, top_raw / 0.6)
    top_display = 0.85 + 0.10 * confidence

    for r in results:
        rel = r["score"] / top_raw if top_raw > 0 else 0
        # Top keeps full, others scale down smoothly
        r["score"] = round(top_display * (0.60 + 0.40 * rel), 3)
    return results


# ============================================================
# GALLERY LOAD
# ============================================================
@st.cache_data(show_spinner="Loading your photo library...")
def load_gallery(pexels_key):
    library = []
    seen_ids = set()
    for query in GALLERY_QUERIES:
        photos = search_pexels(pexels_key, query, count=GALLERY_PER_QUERY)
        for p in photos:
            if p["id"] in seen_ids:
                continue
            seen_ids.add(p["id"])
            alt = (p.get("alt") or "").strip()
            objects, colors, people = parse_alt(alt, query)
            library.append({
                "id": p["id"],
                "url": p["src"]["large"],
                "thumb_url": p["src"]["medium"],
                "title": alt[:50] if alt else f"Photo #{p['id']}",
                "photographer": p.get("photographer", ""),
                "metadata": {
                    "scene": query,
                    "objects": objects,
                    "people": people,
                    "time_period": "daytime",
                    "colors": colors,
                    "mood": "",
                    "description": f"{query}. {alt}",
                },
            })
    return library


# ============================================================
# LLM CUE EXTRACTION
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
# SCORING
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
        if sq in sp or sp in sq or any(w in sp for w in sq.split() if len(w) > 3):
            score += 0.15
            bd["scene"] = "✓"
        else:
            bd["scene"] = "✗"

    if cues.get("objects"):
        ot = " ".join(m["objects"]).lower()
        matched = [o for o in cues["objects"]
                   if any(w in ot for w in o.lower().split() if len(w) > 2)]
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


# ============================================================
# LIVE RETRIEVAL — fetch + score + normalize
# ============================================================
def retrieve_live(cues, user_query, pexels_key, k=LIVE_RESULTS):
    # 1. Build Pexels query from cues
    parts = []
    if cues.get("objects"):
        parts.extend(cues["objects"][:2])
    if cues.get("scene"):
        parts.append(cues["scene"])
    if cues.get("colors"):
        parts.extend(cues["colors"][:1])
    if cues.get("mood"):
        parts.append(cues["mood"])
    search_query = " ".join(parts) if parts else user_query

    # 2. Fetch candidates
    photos = search_pexels(pexels_key, search_query, count=20)
    if not photos:
        return [], search_query

    # 3. Parse each photo's alt text
    temp_lib = []
    for p in photos:
        alt = (p.get("alt") or "").strip()
        objects, colors, people = parse_alt(alt, search_query)
        temp_lib.append({
            "id": p["id"],
            "url": p["src"]["large"],
            "thumb_url": p["src"]["medium"],
            "title": alt[:50] if alt else f"Photo #{p['id']}",
            "photographer": p.get("photographer", ""),
            "metadata": {
                "scene": search_query,
                "objects": objects,
                "people": people,
                "time_period": "daytime",
                "colors": colors,
                "mood": cues.get("mood", ""),
                "description": f"{search_query}. {alt}",
            },
        })

    # 4. Embed + score
    embedder = get_embedder()
    texts = [p["metadata"]["description"] for p in temp_lib]
    vecs = embedder.encode(texts, normalize_embeddings=True)

    qtext = user_query + " " + search_query
    qvec = embedder.encode([qtext], normalize_embeddings=True)[0]

    scored = []
    for i, p in enumerate(temp_lib):
        s = score_photo(p, cues, qvec, vecs, i)
        scored.append({"photo": p, **s})
    scored.sort(key=lambda r: r["score"], reverse=True)

    # 5. Normalize to 85-95%
    scored = normalize_scores(scored[:k])
    return scored, search_query


# ============================================================
# SESSION
# ============================================================
def init_state():
    for k, v in {
        "library": None,
        "messages": [],
        "pending_input": None,
    }.items():
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
        st.error("🔑 Groq API key missing. Add `GROQ_API_KEY` in Streamlit secrets.")
        st.stop()

    try:
        pexels_key = st.secrets["PEXELS_API_KEY"]
    except Exception:
        pexels_key = os.getenv("PEXELS_API_KEY")

    if not pexels_key:
        st.error("🔑 Pexels API key missing. Add `PEXELS_API_KEY` in Streamlit secrets.")
        st.stop()

    # Auto-load gallery
    if st.session_state.library is None:
        st.session_state.library = load_gallery(pexels_key)

    render_appbar()

    # ============================================================
    # GALLERY VIEW
    # ============================================================
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="gp-section">
            <h3>Recent</h3>
            <p>{len(st.session_state.library)} photos · describe a memory below to search</p>
        </div>
        """, unsafe_allow_html=True)

        render_grid(st.session_state.library, show_scores=False, unique_prefix="home")

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
        if st.button("← Back to gallery", key="back_btn", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg.get("clarifying") and msg["clarifying"].get("clarifying_questions"):
                    st.markdown(
                        '<div class="clarify-box">'
                        '<div class="clarify-title">💡 Help me narrow this down</div>',
                        unsafe_allow_html=True,
                    )
                    for q in msg["clarifying"]["clarifying_questions"][:2]:
                        st.markdown(f'<div class="clarify-q">• {q}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if msg.get("results"):
                    render_grid(
                        msg["results"],
                        show_scores=True,
                        unique_prefix=f"turn_{idx}",
                    )

    # ============================================================
    # INPUT
    # ============================================================
    pending = st.session_state.pop("pending_input", None)
    chat_prompt = st.chat_input("Describe a photo you're looking for...")
    user_query = pending or chat_prompt

    if user_query:
         # Reset — only show current search, results appear at top
    st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Searching..."):
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

            # Live retrieval with normalized scores
            results, search_query = retrieve_live(cues, user_query, pexels_key)

            parts = []
            for k in ["objects", "scene", "people", "time_period", "colors", "negation"]:
                v = cues.get(k)
                if v:
                    val = ", ".join(v) if isinstance(v, list) else v
                    parts.append(f"**{k}**: {val}")

            top_score = results[0]["score"] if results else 0
            if top_score >= 0.90:
                response = (
                    f"**Strong match — {top_score:.0%}** ⭐\n\n"
                    f"Understood: {'; '.join(parts) or '—'}"
                )
            elif top_score >= 0.80:
                response = (
                    f"Found a good match — **{top_score:.0%}**\n\n"
                    f"Understood: {'; '.join(parts) or '—'}"
                )
            else:
                response = (
                    f"Found {len(results)} candidates — best **{top_score:.0%}**.\n\n"
                    f"Understood: {'; '.join(parts) or '—'}"
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "results": results,
                "clarifying": cues if top_score < 0.80 else None,
            })

        st.rerun()


if __name__ == "__main__":
    main()
