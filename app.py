"""Describe it: conversational retrieval for vaguely remembered photos (Google Photos-style MVP)."""
import json
import os
import re
import time

import numpy as np
import requests
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Photos · Describe it", page_icon="📸",
                   layout="centered", initial_sidebar_state="collapsed")

# ============================================================
# CONFIG
# ============================================================
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")   # fast; switch to gpt-oss-120b for quality
POOL_PER_QUERY = 30
SHOW_MAX = 6
HIGH, POSSIBLE = 0.62, 0.40          # honest thresholds on 0-1 cue coverage
GALLERY = [  # (section title, subtitle, pexels query)
    ("Today", "San Francisco", "friends cafe"),
    ("Yesterday", "Wed, May 22", "beach sunset friends"),
    ("May 2024", "24 items", "family outdoor"),
    ("April 2024", "Spring break", "school students uniform"),
    ("March 2024", "Trip", "mountain hiking"),
    ("February 2024", "", "birthday party"),
]
EXAMPLES = [
    "Me and friends at a beach at sunset",
    "School friends in uniform on the school ground",
    "Me alone at a café in a red t-shirt",
    "The medicine strip I photographed when I was sick",
]

# ============================================================
# STYLE (Google Photos / Material 3, from DESIGN.md)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,wght@8..144,400;8..144,500;8..144,600&display=swap');
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.stApp, [data-testid="stAppViewContainer"] {background:#FAF9FD !important; color:#1A1B1E !important;}
[data-testid="stMainBlockContainer"], .main .block-container {
  max-width:430px !important; padding:14px 14px 120px !important; margin:0 auto !important;}
.stMarkdown, .stMarkdown p, label, summary, [data-testid="stExpander"] *, [data-testid="stCaptionContainer"] {color:#1A1B1E !important;}
[data-testid="stBottom"], [data-testid="stBottom"] > div {background:#FAF9FD !important;}
[data-testid="stBottomBlockContainer"] {max-width:430px !important; margin:0 auto !important; padding:10px 14px 16px !important;}
[data-testid="stChatInput"] {background:#F1F3F4 !important; border:1px solid #DADCE0 !important; border-radius:28px !important;}
[data-testid="stChatInput"] textarea {color:#1A1B1E !important; background:transparent !important;}
button[kind="secondary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-pills"] {
  background:#FFFFFF !important; color:#1A73E8 !important; border:1px solid #DADCE0 !important;}
[data-testid="stBaseButton-pillsActive"] {background:#D3E3FD !important; color:#041E49 !important; border:0 !important;}
.why {display:flex; flex-wrap:wrap; gap:4px; margin:2px 0 6px;}
.why span {font-size:14px; border-radius:8px; padding:2px 8px;}
.why .ok {background:#C4EED0; color:#0D652D;} .why .no {background:#FFDAD6; color:#93000A;} .why .na {background:#E3E2E6; color:#414754;}
.hint {font-size:14px; color:#5F6368; margin:-2px 0 6px;}
html, body, .stMarkdown, button, input, textarea, p, span, div {font-family:'Roboto Flex', Roboto, Arial, sans-serif;}
.appbar {display:flex; align-items:center; gap:10px; padding:4px 2px 10px; font-size:22px; color:#1A1B1E;}
.dots {display:grid; grid-template-columns:repeat(2,7px); gap:2px;}
.dots i {width:7px; height:7px; border-radius:50%; display:block;}
.chiprow {display:flex; gap:8px; overflow-x:auto; padding:2px 0 6px;}
.chip {border:1px solid #DADCE0; border-radius:8px; padding:5px 12px; font-size:14px; white-space:nowrap; color:#202124; background:#fff;}
.sec {display:flex; align-items:baseline; gap:8px; margin:18px 2px 8px; font-size:18px; font-weight:500; color:#1A1B1E;}
.sec small, .meta {font-size:14px; color:#5F6368; font-weight:400;}
.grid3 {display:grid; grid-template-columns:repeat(3,1fr); gap:3px;}
.grid3 img {width:100%; aspect-ratio:1; object-fit:cover; display:block;}
.bubble {margin:6px 0 6px auto; max-width:85%; width:fit-content; background:#1A73E8; color:#fff;
  border-radius:20px 20px 4px 20px; padding:12px 16px; font-size:16px; line-height:1.4;}
.card {background:#F1F3F4; border-radius:20px; padding:14px 16px; margin:10px 0 6px;}
.card h4 {margin:0 0 2px; font-size:18px; font-weight:500; color:#1A1B1E; padding:0;}
.turn {font-size:14px; color:#5F6368; padding:2px;}
.ph {position:relative; border-radius:14px; overflow:hidden; margin-bottom:4px;}
.ph img {width:100%; aspect-ratio:1; object-fit:cover; display:block;}
.badge {position:absolute; top:8px; left:8px; border-radius:999px; padding:3px 10px; font-size:14px; font-weight:600;}
.high {background:#C4EED0; color:#0D652D;} .possible {background:#FEF7E0; color:#7A4100;} .low {background:#E3E2E6; color:#414754;}
.found {background:#2F3033; color:#fff; border-radius:16px; padding:12px 16px; font-size:16px; margin:10px 0;}
.hero {border-radius:24px; overflow:hidden;} .hero img {width:100%; display:block;}
.stButton > button {border-radius:999px; font-weight:500; min-height:40px; font-size:14px;}
button[kind="primary"], [data-testid="stBaseButton-primary"] {background:#1A73E8 !important; border-color:#1A73E8 !important;}
[data-testid="stExpander"] summary p {font-size:14px;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# RESOURCES
# ============================================================
def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


@st.cache_resource
def get_groq():
    key = secret("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


@st.cache_resource(show_spinner="Warming up search…")
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_data(show_spinner=False, ttl=86400)
def search_pexels(api_key, query, count=POOL_PER_QUERY):
    try:
        r = requests.get("https://api.pexels.com/v1/search", timeout=10,
                         headers={"Authorization": api_key},
                         params={"query": query, "per_page": count})
        r.raise_for_status()
        return [{"id": p["id"], "thumb": p["src"]["medium"], "large": p["src"]["large"],
                 "alt": (p.get("alt") or "").strip()} for p in r.json().get("photos", [])]
    except Exception:
        return []


# ============================================================
# CLUE EXTRACTION (one LLM call per chat turn)
# ============================================================
EMPTY = {"people": [], "alone": False, "activity": "", "scene": "", "setting": "",
         "clothing": [], "colors": [], "objects": [], "time_period": "", "mood": ""}

TURN_PROMPT = """You help a user find an old photo they only vaguely remember. You keep structured clues across a chat.
Given CURRENT_CLUES, LAST_QUESTION and the user's NEW_MESSAGE, return ONLY JSON:
{"clues": {"people": [], "alone": false, "activity": "", "scene": "", "setting": "indoors|outdoors|",
  "clothing": [], "colors": [], "objects": [], "time_period": "", "mood": ""},
 "question": {"text": "one short question that best narrows the search", "options": ["2-3 short answers"]},
 "rephrases": ["3 short alternative ways to describe the photo"]}
Rules:
- Merge: keep existing clues unless NEW_MESSAGE changes or removes them.
- If NEW_MESSAGE answers LAST_QUESTION (e.g. "Outdoors"), store it in the right field.
- "me alone" / "by myself" -> people ["me"], alone true.
- Keep colour with clothing: "red t-shirt" -> clothing ["red t-shirt"], colors ["red"].
- Place types go to scene (cafe, beach, school ground). Events go to activity (birthday, trip, ramp walk).
- Ask only about something NOT yet known (setting, who else was there, activity, time). Add "Not sure" as an option.
- Every value 1-3 words."""


def extract_clues(current, message, last_q):
    client = get_groq()
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": TURN_PROMPT},
                      {"role": "user", "content": json.dumps(
                          {"CURRENT_CLUES": current, "LAST_QUESTION": last_q, "NEW_MESSAGE": message})}])
        out = json.loads(r.choices[0].message.content)
    except Exception:
        out = {"clues": {**current, "objects": current["objects"] +
                         [w for w in re.findall(r"[a-z]+", message.lower()) if len(w) > 3]}}
    clues = {**EMPTY, **{k: v for k, v in (out.get("clues") or {}).items() if k in EMPTY}}
    for k in ("people", "clothing", "colors", "objects"):
        clues[k] = [str(x) for x in (clues[k] or []) if str(x).strip()][:4]
    clues["setting"] = clues["setting"] if clues["setting"] in ("indoors", "outdoors") else ""
    q = out.get("question") or {}
    question = {"text": q.get("text", ""), "options": (q.get("options") or [])[:3]} if q.get("text") else None
    return clues, question, (out.get("rephrases") or [])[:3]


# ============================================================
# HONEST SCORING — photo alt text only (the query never leaks into photo metadata)
# ============================================================
SYN = {
    "cafe": ["cafe", "café", "coffee", "restaurant", "bistro"], "café": ["cafe", "café", "coffee", "restaurant"],
    "beach": ["beach", "sea", "ocean", "shore", "coast"], "school": ["school", "classroom", "campus", "student"],
    "park": ["park", "garden", "grass"], "party": ["party", "celebration", "balloon"],
    "birthday": ["birthday", "cake", "party", "balloon"], "wedding": ["wedding", "bride", "groom"],
    "mountain": ["mountain", "hill", "peak"], "trip": ["travel", "trip", "tourist", "vacation"],
    "t-shirt": ["t-shirt", "tshirt", "shirt", "top", "tee"], "tshirt": ["t-shirt", "shirt", "top"],
    "shirt": ["shirt", "t-shirt", "top"], "dress": ["dress", "gown", "saree"], "uniform": ["uniform", "student"],
    "medicine": ["medicine", "pill", "pills", "tablet", "capsule", "medication", "drug"],
    "sunset": ["sunset", "dusk", "golden hour", "evening"], "hiking": ["hiking", "hike", "trek", "trail"],
}
PEOPLE = {
    "me": ["man", "woman", "person", "boy", "girl", "guy", "lady"],
    "friend": ["friend", "friends", "group", "people", "together"],
    "friends": ["friend", "friends", "group", "people", "together"],
    "family": ["family", "parents", "children", "kids", "mother", "father"],
    "dad": ["man", "father", "dad"], "mom": ["woman", "mother", "mom"],
    "daughter": ["girl", "daughter", "child"], "son": ["boy", "son", "child"],
}
GROUP = ["friends", "group", "people", "crowd", "couple", "family", "together", "team"]
OUTDOOR = ["outdoor", "outside", "street", "park", "garden", "beach", "sky", "terrace", "sidewalk", "field", "mountain", "sea"]
INDOOR = ["indoor", "inside", "room", "interior", "home", "office", "classroom", "kitchen"]
STOP = {"the", "and", "with", "wearing", "was", "were", "my", "our", "a", "an", "of", "at", "in", "on"}


def hit(phrase, text, mode="all"):
    words = [w for w in re.findall(r"[a-zé\-]+", phrase.lower()) if len(w) > 2 and w not in STOP]
    if not words:
        return False
    found = [any(a in text for a in SYN.get(w, [w])) for w in words]
    return all(found) if mode == "all" else any(found)


def score_photo(photo, clues, sem):
    t = photo["alt"].lower()
    earned = 0.18 * min(1.0, max(0.0, (sem - 0.15) / 0.45))
    possible, checks = 0.18, []

    def add(label, ok, w):
        nonlocal earned, possible
        possible += w
        earned += w if ok else 0
        checks.append((label, ok))

    for p in clues["people"]:
        add(f"Person: {p}", any(a in t for a in PEOPLE.get(p.lower(), [p.lower()])), 0.20 / len(clues["people"]))
    if clues["alone"]:
        add("Just you, no group", not any(g in t for g in GROUP), 0.10)
    if clues["scene"]:
        add(f"Place: {clues['scene']}", hit(clues["scene"], t, "any"), 0.15)
    if clues["setting"]:
        words = OUTDOOR if clues["setting"] == "outdoors" else INDOOR
        add(f"Setting: {clues['setting']}", any(w in t for w in words), 0.08)
    if clues["activity"]:
        add(f"Activity: {clues['activity']}", hit(clues["activity"], t, "any"), 0.12)
    for c in clues["clothing"]:
        add(f"Clothing: {c}", hit(c, t, "all"), 0.12 / len(clues["clothing"]))
    for c in clues["colors"]:
        add(f"Colour: {c}", c.lower() in t, 0.05 / len(clues["colors"]))
    for o in clues["objects"]:
        add(f"Object: {o}", hit(o, t, "all"), 0.10 / len(clues["objects"]))
    if clues["time_period"]:
        checks.append((f"When: {clues['time_period']} (demo photos have no date)", None))
    return earned / possible, checks


def label(score):
    if score >= HIGH:
        return "high", "✓ High match"
    if score >= POSSIBLE:
        return "possible", "? Possible"
    return "low", "Low"


# ============================================================
# RETRIEVAL
# ============================================================
DAY = ["day", "sunny", "daylight", "morning", "afternoon", "bright"]
NIGHT = ["night", "evening", "sunset", "dusk", "dark", "lights"]


def split_question(ranked, c):
    """Ask about the attribute that splits the closest candidates most evenly (grounded, no extra LLM call)."""
    top = [st.session_state.pool[r["id"]]["alt"].lower() for r in ranked[:10]]
    known_group = c["alone"] or len(c["people"]) > 1 or any(p.lower() in ("friends", "family") for p in c["people"])
    options = []
    if not c["setting"]:
        options.append(("Was it indoors or outdoors?", "Outdoors", OUTDOOR, "Indoors", INDOOR))
    if not known_group:
        options.append(("Were you alone or with others?", "With others", GROUP, "Just me", None))
    if not c["time_period"]:
        options.append(("Was it day or evening?", "Daytime", DAY, "Evening", NIGHT))
    best, best_bal = None, 1
    for text, a_lbl, a_words, b_lbl, b_words in options:
        a = sum(any(w in t for w in a_words) for t in top)
        b = sum((any(w in t for w in b_words) if b_words else not any(w in t for w in a_words)) for t in top)
        if min(a, b) > best_bal:
            best, best_bal = {"text": text, "options": [a_lbl, b_lbl, "Not sure"], "grounded": True}, min(a, b)
    return best


def build_query(c):
    parts = []
    if c["people"]:
        parts.append("person alone" if c["alone"] else " ".join("person" if p.lower() == "me" else p for p in c["people"][:2]))
    parts += [c["activity"], c["scene"], c["clothing"][0] if c["clothing"] else "",
              "" if c["clothing"] or not c["objects"] else c["objects"][0],
              "outdoor" if c["setting"] == "outdoors" else ""]
    return " ".join(p for p in parts if p).strip() or "people"


def add_to_pool(photos):
    pool, emb = st.session_state.pool, st.session_state.emb
    new = [p for p in photos if p["id"] not in pool and p["alt"]]
    for p in new:
        pool[p["id"]] = p
    if new:
        vecs = get_embedder().encode([p["alt"] for p in new], normalize_embeddings=True)
        for p, v in zip(new, vecs):
            emb[p["id"]] = v


def rank(clues, anchor_id=None):
    pool, emb = st.session_state.pool, st.session_state.emb
    ids = list(pool)
    if not ids:
        return []
    mat = np.stack([emb[i] for i in ids])
    qtext = " ".join(st.session_state.user_msgs[-3:])
    sem = mat @ get_embedder().encode([qtext], normalize_embeddings=True)[0]
    sim = mat @ emb[anchor_id] if anchor_id else None
    out = []
    for k, pid in enumerate(ids):
        if pid == anchor_id:
            continue
        s, checks = score_photo(pool[pid], clues, float(sem[k]))
        if anchor_id:  # "Find similar": blend look-alike similarity with the remembered clues
            look = min(1.0, max(0.0, (float(sim[k]) - 0.2) / 0.5))
            s = 0.5 * s + 0.5 * look
            checks = [("Looks like the photo you picked", look >= 0.5)] + checks
        out.append({"id": pid, "score": s, "checks": checks})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def snapshot():
    keys = ("clues", "results", "question", "rephrases", "matching")
    return {k: st.session_state[k] for k in keys}


def apply_results(ranked):
    s = st.session_state
    s.prev_matching = s.matching
    s.matching = sum(r["score"] >= POSSIBLE for r in ranked)
    shown = [r for r in ranked if r["score"] >= POSSIBLE][:SHOW_MAX]
    s.results = shown or ranked[:3]
    s.low_conf = not shown


def run_turn(msg, pexels_key):
    s = st.session_state
    if s.start is None:
        s.start = time.time()
    s.history.append(snapshot())
    s.user_msgs.append(msg)
    last_q = s.question["text"] if s.question else ""
    s.clues, s.question, s.rephrases = extract_clues(s.clues, msg, last_q)
    add_to_pool(search_pexels(pexels_key, build_query(s.clues)))
    ranked = rank(s.clues)
    apply_results(ranked)
    s.question = split_question(ranked, s.clues) or s.question
    s.turns.append(msg)
    s.clue_ver += 1


def find_similar(pid, pexels_key):
    s = st.session_state
    s.history.append(snapshot())
    words = [w for w in re.findall(r"[a-z]+", s.pool[pid]["alt"].lower()) if w not in STOP][:6]
    add_to_pool(search_pexels(pexels_key, " ".join(words)))
    apply_results(rank(s.clues, anchor_id=pid))
    s.turns.append("Find similar to the photo you picked")
    s.question = None


def go_back():
    s = st.session_state
    if s.history:
        for k, v in s.history.pop().items():
            s[k] = v
        s.turns = s.turns[:-1]
        s.clue_ver += 1


# ============================================================
# STATE
# ============================================================
def init_state():
    defaults = {"screen": "home", "clues": dict(EMPTY), "results": [], "question": None, "rephrases": [],
                "matching": 0, "prev_matching": 0, "low_conf": False, "history": [], "turns": [],
                "user_msgs": [], "pool": {}, "emb": {}, "start": None, "found": None, "clue_ver": 0,
                "pending": None, "log": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_search():
    for k in ("clues", "results", "question", "rephrases", "matching", "prev_matching", "low_conf",
              "history", "turns", "user_msgs", "start", "found"):
        del st.session_state[k]
    init_state()


# ============================================================
# UI
# ============================================================
def appbar(title="Photos"):
    st.markdown(f"""<div class="appbar"><div class="dots"><i style="background:#4285F4"></i><i style="background:#EA4335"></i>
    <i style="background:#FBBC04"></i><i style="background:#34A853"></i></div><span>{title}</span></div>""",
                unsafe_allow_html=True)


def clue_labels(c):
    items = [("people", p, f"👤 {p}") for p in c["people"]]
    if c["alone"]:
        items.append(("alone", True, "👤 alone"))
    if c["scene"]:
        items.append(("scene", c["scene"], f"📍 {c['scene']}"))
    if c["setting"]:
        items.append(("setting", c["setting"], f"☀️ {c['setting']}" if c["setting"] == "outdoors" else f"🏠 {c['setting']}"))
    if c["activity"]:
        items.append(("activity", c["activity"], f"🎉 {c['activity']}"))
    items += [("clothing", x, f"👕 {x}") for x in c["clothing"]]
    items += [("colors", x, f"🎨 {x}") for x in c["colors"]]
    items += [("objects", x, f"🔎 {x}") for x in c["objects"]]
    if c["time_period"]:
        items.append(("time_period", c["time_period"], f"🕒 {c['time_period']}"))
    return items


def remove_clue(field, value):
    c = st.session_state.clues
    if isinstance(c[field], list):
        c[field] = [x for x in c[field] if x != value]
    else:
        c[field] = False if field == "alone" else ""


def screen_home(pexels_key):
    appbar("Google Photos")
    st.markdown('<div class="chiprow">' + "".join(f'<span class="chip">{x}</span>' for x in
                ["♡ Favorites", "▶ Videos", "⛰ Trips", "▭ Screenshots"]) + "</div>", unsafe_allow_html=True)
    if st.button("✨ Can't find a photo? Describe it", type="primary", use_container_width=True):
        st.session_state.screen = "chat"
        st.rerun()
    for title, sub, q in GALLERY:
        photos = search_pexels(pexels_key, q, 9)[:6]
        if not photos:
            continue
        imgs = "".join(f'<img src="{p["thumb"]}" loading="lazy" alt="{p["alt"][:60]}">' for p in photos)
        st.markdown(f'<div class="sec">{title}<small>{sub}</small></div><div class="grid3">{imgs}</div>',
                    unsafe_allow_html=True)


def why_chips(checks):
    out = []
    for lbl, ok in checks:
        short = lbl.split(": ", 1)[-1].replace(" (demo photos have no date)", "")
        cls, sym = ("ok", "✓") if ok else (("no", "✗") if ok is False else ("na", "–"))
        out.append(f'<span class="{cls}">{sym} {short}</span>')
    return '<div class="why">' + "".join(out) + "</div>"


def photo_card(r, pexels_key, i):
    s = st.session_state
    p = s.pool[r["id"]]
    cls, text = label(r["score"])
    st.markdown(f'<div class="ph"><img src="{p["thumb"]}" alt="{p["alt"][:80]}"><span class="badge {cls}">'
                f'{text} · {r["score"]:.0%}</span></div>', unsafe_allow_html=True)
    st.markdown(why_chips(r["checks"]), unsafe_allow_html=True)
    if st.button("✓ This is it", key=f"pick_{i}_{r['id']}", type="primary", use_container_width=True):
        s.found = r
        s.screen = "found"
        s.log.append({"turns": len(s.turns), "seconds": round(time.time() - (s.start or time.time())),
                      "score": round(r["score"], 2), "query": " | ".join(s.user_msgs)})
        st.rerun()
    if st.button("Similar photos", key=f"sim_{i}_{r['id']}", use_container_width=True):
        with st.spinner("Finding similar photos…"):
            find_similar(r["id"], pexels_key)
        st.rerun()


def screen_chat(pexels_key):
    s = st.session_state
    c1, c2 = st.columns([1, 3])
    if c1.button("← Photos"):
        s.screen = "home"
        st.rerun()
    c2.markdown('<div class="appbar" style="padding-top:6px">Describe it</div>', unsafe_allow_html=True)

    if not s.turns:
        st.markdown('<div class="card"><h4>Tell me everything you remember</h4><div class="meta">Who was there, '
                    'what they wore, where it was, the mood. Don\'t simplify it.</div></div>', unsafe_allow_html=True)
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                s.pending = ex
                st.rerun()
    else:
        for k, t in enumerate(s.turns[:-1]):
            st.markdown(f'<div class="turn">↺ Turn {k + 1}: “{t}”</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">{s.turns[-1]}</div>', unsafe_allow_html=True)

        funnel = (f"{s.prev_matching} → {s.matching} matching photos" if len(s.turns) > 1
                  else f"{s.matching} matching photos")
        head = "I'm not sure yet" if s.low_conf else "Here's what I understood"
        st.markdown(f'<div class="card"><h4>{head}</h4><div class="meta">{funnel} · tap a clue to remove it</div></div>',
                    unsafe_allow_html=True)
        items = clue_labels(s.clues)
        if items:
            labels = [x[2] for x in items]
            kept = st.pills("Clues", labels, selection_mode="multi", default=labels,
                            key=f"clues_{s.clue_ver}", label_visibility="collapsed")
            if kept is not None and len(kept) < len(labels):
                s.history.append(snapshot())
                for field, value, lbl in items:
                    if lbl not in kept:
                        remove_clue(field, value)
                apply_results(rank(s.clues))
                s.turns.append("Removed a clue")
                s.clue_ver += 1
                st.rerun()
        if s.history and st.button("↶ Go back a step"):
            go_back()
            st.rerun()

        if s.low_conf and s.rephrases:
            st.markdown('<div class="meta">No strong match. Try describing it another way:</div>', unsafe_allow_html=True)
            for k, rp in enumerate(s.rephrases):
                if st.button(rp, key=f"rp_{s.clue_ver}_{k}", use_container_width=True):
                    s.pending = rp
                    st.rerun()

        if s.question and s.question.get("options"):
            why = ("Picked to split your closest matches" if s.question.get("grounded")
                   else "Answer to narrow the search")
            st.markdown(f'<div class="card"><div class="meta">HELP NARROW IT DOWN</div><h4>{s.question["text"]}</h4>'
                        f'<div class="meta">{why}</div></div>', unsafe_allow_html=True)
            qcols = st.columns(len(s.question["options"]))
            for k, opt in enumerate(s.question["options"]):
                if qcols[k].button(opt, key=f"q_{s.clue_ver}_{k}", use_container_width=True):
                    s.pending = opt
                    st.rerun()

        st.markdown('<div class="sec">Best candidates<small>sorted by match</small></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, r in enumerate(s.results):
            with cols[i % 2]:
                photo_card(r, pexels_key, i)

    msg = st.chat_input("Add another detail… (who, where, what they wore)" if s.turns
                        else "Describe what you remember…")
    msg = s.pending or msg
    if msg:
        s.pending = None
        with st.spinner("Searching your memories…"):
            run_turn(msg, pexels_key)
        st.rerun()


def screen_found():
    s = st.session_state
    r = s.found
    p = s.pool[r["id"]]
    appbar("Photo")
    secs = s.log[-1]["seconds"] if s.log else 0
    st.markdown(f'<div class="hero"><img src="{p["large"]}" alt="{p["alt"][:80]}"></div>'
                f'<div class="found">✓ Found it! Saved to Favourites<br><span style="font-size:14px;opacity:.85">'
                f'{len(s.turns)} turn(s) · {secs}s</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Why this photo</div>' + why_chips(r["checks"]), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("← Not it, go back", use_container_width=True):
        s.screen = "chat"
        st.rerun()
    if c2.button("Find another photo", type="primary", use_container_width=True):
        reset_search()
        s.screen = "chat"
        st.rerun()
    if s.log:
        st.download_button("Download test log (CSV)", data="turns,seconds,score,query\n" + "\n".join(
            f'{x["turns"]},{x["seconds"]},{x["score"]},"{x["query"]}"' for x in s.log),
            file_name="retrieval_test_log.csv", use_container_width=True)


def main():
    init_state()
    pexels_key = secret("PEXELS_API_KEY")
    if get_groq() is None or not pexels_key:
        st.error("Add GROQ_API_KEY and PEXELS_API_KEY in Streamlit secrets.")
        st.stop()
    {"home": lambda: screen_home(pexels_key), "chat": lambda: screen_chat(pexels_key),
     "found": screen_found}[st.session_state.screen]()


main()
