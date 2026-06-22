import os
import streamlit as st
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="FrameFM", page_icon="📻", layout="centered")

# ── Sidebar: credentials & settings ──────────────────────────────────────────
with st.sidebar:
    st.header("Spotify Credentials")
    client_id = st.text_input("Client ID", type="password",
                              value=os.getenv("SPOTIFY_CLIENT_ID", ""))
    client_secret = st.text_input("Client Secret", type="password",
                                  value=os.getenv("SPOTIFY_CLIENT_SECRET", ""))
    st.divider()
    num_songs = st.slider("Number of songs", min_value=3, max_value=20, value=10)

# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource(show_spinner="Loading image captioning model…")
def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


@st.cache_resource(show_spinner="Loading emotion classifier…")
def load_classifier():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )


def caption_image(image: Image.Image) -> str:
    import torch
    processor, model, device = load_blip()
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True)


MOOD_LABELS = [
    "happy", "sad", "energetic", "calm", "romantic", "dark", "dreamy",
    "nostalgic", "angry", "peaceful", "mysterious", "epic", "melancholic",
    "uplifting", "chill",
]

MOOD_TO_SPOTIFY_ATTRS = {
    "happy":       dict(valence=(0.7, 1.0),  energy=(0.5, 1.0)),
    "sad":         dict(valence=(0.0, 0.35), energy=(0.0, 0.5)),
    "energetic":   dict(energy=(0.7, 1.0),   tempo=(120, 200)),
    "calm":        dict(energy=(0.0, 0.4),   valence=(0.3, 0.7)),
    "romantic":    dict(valence=(0.5, 0.9),  energy=(0.2, 0.6)),
    "dark":        dict(valence=(0.0, 0.4),  energy=(0.4, 0.8)),
    "dreamy":      dict(valence=(0.4, 0.8),  energy=(0.1, 0.45)),
    "nostalgic":   dict(valence=(0.3, 0.7),  energy=(0.2, 0.6)),
    "angry":       dict(valence=(0.0, 0.4),  energy=(0.7, 1.0)),
    "peaceful":    dict(energy=(0.0, 0.35),  valence=(0.4, 0.8)),
    "mysterious":  dict(valence=(0.2, 0.6),  energy=(0.2, 0.6)),
    "epic":        dict(energy=(0.7, 1.0),   valence=(0.4, 0.9)),
    "melancholic": dict(valence=(0.1, 0.4),  energy=(0.1, 0.5)),
    "uplifting":   dict(valence=(0.6, 1.0),  energy=(0.5, 0.9)),
    "chill":       dict(energy=(0.1, 0.5),   valence=(0.4, 0.8)),
}


def top_moods(caption: str, top_k: int = 3) -> list[str]:
    classifier = load_classifier()
    result = classifier(caption, MOOD_LABELS, multi_label=True)
    ranked = sorted(zip(result["labels"], result["scores"]),
                    key=lambda x: x[1], reverse=True)
    return [label for label, _ in ranked[:top_k]]


def spotify_client(cid: str, secret: str) -> spotipy.Spotify:
    auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    return spotipy.Spotify(auth_manager=auth)


def get_songs(sp: spotipy.Spotify, moods: list[str], caption: str, n: int):
    query_terms = caption.split()[:5]
    query_terms.append(moods[0])
    query = " ".join(query_terms)

    attrs = MOOD_TO_SPOTIFY_ATTRS.get(moods[0], {})
    target_valence = sum(attrs.get("valence", (0.5, 0.5))) / 2 if "valence" in attrs else None
    target_energy  = sum(attrs.get("energy",  (0.5, 0.5))) / 2 if "energy"  in attrs else None

    results = sp.search(q=query, type="track", limit=50)
    tracks = results["tracks"]["items"]

    if not tracks:
        return []

    ids = [t["id"] for t in tracks if t.get("id")]
    features = sp.audio_features(ids[:50])
    feat_map = {f["id"]: f for f in features if f}

    def score(track):
        f = feat_map.get(track["id"])
        if not f:
            return 0
        s = 0
        if target_valence is not None:
            s += 1 - abs(f.get("valence", 0.5) - target_valence)
        if target_energy is not None:
            s += 1 - abs(f.get("energy", 0.5) - target_energy)
        return s

    ranked = sorted(tracks, key=score, reverse=True)
    return ranked[:n]


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📻 FrameFM")
st.caption("Upload a photo and get a playlist that matches the vibe.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    if not client_id or not client_secret:
        st.warning("Enter your Spotify credentials in the sidebar to continue.")
        st.stop()

    with st.spinner("Analyzing image…"):
        caption = caption_image(image)

    st.markdown(f"**Image description:** _{caption}_")

    with st.spinner("Detecting vibe…"):
        moods = top_moods(caption)

    st.markdown(f"**Detected moods:** {' · '.join(f'`{m}`' for m in moods)}")

    with st.spinner("Finding songs…"):
        try:
            sp = spotify_client(client_id, client_secret)
            songs = get_songs(sp, moods, caption, num_songs)
        except Exception as e:
            st.error(f"Spotify error: {e}")
            st.stop()

    if not songs:
        st.warning("No songs found — try a different image.")
        st.stop()

    st.divider()
    st.subheader(f"🎧 {len(songs)} songs for this vibe")

    for i, track in enumerate(songs, 1):
        artists = ", ".join(a["name"] for a in track["artists"])
        name    = track["name"]
        url     = track["external_urls"].get("spotify", "#")
        album   = track["album"]["name"]
        art     = track["album"]["images"][-1]["url"] if track["album"]["images"] else None

        cols = st.columns([1, 6])
        if art:
            cols[0].image(art, width=55)
        cols[1].markdown(
            f"**{i}. [{name}]({url})**  \n{artists} — *{album}*"
        )
