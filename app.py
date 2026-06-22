import os
import streamlit as st
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
load_dotenv()

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

MOOD_TO_GENRE = {
    "happy":       "pop",
    "sad":         "sad",
    "energetic":   "dance",
    "calm":        "ambient",
    "romantic":    "romance",
    "dark":        "dark",
    "dreamy":      "dream pop",
    "nostalgic":   "indie",
    "angry":       "metal",
    "peaceful":    "acoustic",
    "mysterious":  "alternative",
    "epic":        "epic",
    "melancholic": "melancholic",
    "uplifting":   "uplifting",
    "chill":       "chill",
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
    genre = MOOD_TO_GENRE.get(moods[0], moods[0])
    query_terms = caption.split()[:4]
    query = " ".join(query_terms) + f" {genre}"

    results = sp.search(q=query, type="track", limit=n)
    tracks = results["tracks"]["items"]

    if not tracks:
        # fallback: search by mood/genre only
        results = sp.search(q=genre, type="track", limit=n)
        tracks = results["tracks"]["items"]

    return sorted(tracks, key=lambda t: t.get("popularity", 0), reverse=True)[:n]


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
