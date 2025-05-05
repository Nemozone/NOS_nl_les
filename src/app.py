import streamlit as st
from tools import *
import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# Hard‑coded channel we want to show on startup
CHANNEL_ID = "UCch2JvY2ZSwcjf5gb93HGQw"

def get_top_videos(channel_id: str, max_results: int = 5):
    """
    Return a list with the `max_results` most‑recent uploads from the given
    YouTube channel.  Each element is a dict: {"title", "video_id", "thumbnail"}.
    This uses only 2 quota units (channels.list + playlistItems.list).
    """
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY missing (set in .env or Streamlit Secrets)")

    # 1 · get the channel's uploads playlist ID (1 unit)
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params=dict(
            part="contentDetails",
            id=channel_id,
            key=YOUTUBE_API_KEY,
        ),
        timeout=10,
    )
    resp.raise_for_status()
    uploads_id = resp.json()["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # 2 · fetch the latest uploads (playlistItems are returned newest‑first) (1 unit)
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        params=dict(
            part="snippet",
            playlistId=uploads_id,
            maxResults=max_results,
            key=YOUTUBE_API_KEY,
        ),
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json()["items"]

    videos = [
        {
            "title": it["snippet"]["title"],
            "video_id": it["snippet"]["resourceId"]["videoId"],
            "thumbnail": (
                it["snippet"]["thumbnails"].get("high")
                or it["snippet"]["thumbnails"].get("medium")
                or it["snippet"]["thumbnails"]["default"]
            )["url"],
        }
        for it in items
    ]
    return videos

def main():
    st.set_page_config(layout="wide")
    st.title("NOS Journal Dutch Learning App")
    st.write("This app extracts the transcript from NOS Journaal in Makkelijke Taal and provides language learning exercises.")

    # Initialize session state variables
    if 'video_shown' not in st.session_state:
        st.session_state.video_shown = False
    if 'url' not in st.session_state:
        st.session_state.url = ''
    if 'videos' not in st.session_state:
        st.session_state.videos = []
    if 'video_titles' not in st.session_state:
        st.session_state.video_titles = []

    # ------------------------------------------------------------------
    # Load top 5 videos for the predefined channel on first run
    # ------------------------------------------------------------------
    if "videos_initialized" not in st.session_state:
        with st.spinner("Loading top videos…"):
            st.session_state.videos = get_top_videos(CHANNEL_ID)
            st.session_state.video_titles = [v["title"] for v in st.session_state.videos]
            st.session_state.videos_initialized = True

    # Display thumbnails for selection
    if st.session_state.videos:
        st.subheader("Pick a video to analyse:")
        cols = st.columns(len(st.session_state.videos))
        for idx, col in enumerate(cols):
            video = st.session_state.videos[idx]
            col.image(video["thumbnail"])
            col.caption(video["title"])
            if col.button("Select", key=video["video_id"]):
                st.session_state.url = f"https://www.youtube.com/watch?v={video['video_id']}"
                st.session_state.selected_title = video["title"]
                st.session_state.video_shown = True
                st.rerun()

    # Display the video if the flag is set
    if st.session_state.video_shown:
        st.video(st.session_state.url)

        # Button to generate the Dutch lesson
        if st.button("Generate Dutch Lesson"):
            with st.spinner("Retrieving transcript..."):
                video_id = parse_url(st.session_state.url)
                transcript = get_text_from_video(video_id)
                transcript_str = "  \n".join(transcript)
                chunks = create_chunks(transcript)
            # -------- side‑by‑side transcript + translation ------------------

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Transcript")
                with st.expander("View Transcript"):
                    st.markdown(transcript_str)
            
            with col2:
                st.subheader("English Translation")
                with st.expander("View English Translation"):
                    # placeholder for live updates
                    placeholder = st.empty()
                    translation_acc = ""

                    # stream the translation line‑by‑line
                    for partial in english_translate_stream_lines(transcript):
                        translation_acc = partial
                        placeholder.markdown(partial + " ▌")  # typing cursor

                    # final update without cursor
                    placeholder.markdown(translation_acc)
                    english_translation = translation_acc  # keep if needed later
            # -----------------------------------------------------------------
            
            
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Vocabulary List:")
                with st.expander("View Vocabulary List"):
                    placeholder_v = st.empty()
                    vocab_acc = ""
                    for partial in get_vocab_stream(chunks):
                        vocab_acc = partial
                        placeholder_v.markdown(partial + " ▌")
                    # final update without cursor
                    placeholder_v.markdown(vocab_acc)
                    vocab_list = vocab_acc
            with col4:
                st.subheader("Grammar Points:")
                with st.expander("View Grammar Points"):
                    placeholder_g = st.empty()
                    grammar_acc = ""
                    for partial in extract_grammar_stream(chunks):
                        grammar_acc = partial
                        placeholder_g.markdown(partial + " ▌")
                    # final update without cursor
                    placeholder_g.markdown(grammar_acc)
                    grammar_points = grammar_acc
            # -----------------------------------------------------------------
            # Generate exercises based on vocabulary and grammar points
            st.subheader("Exercises:")
            with st.expander("View Exercises"):
                placeholder_e = st.empty()
                exercises_acc = ""
                for partial in create_exercises_stream(vocab_list, grammar_points):
                    exercises_acc = partial
                    placeholder_e.markdown(partial + " ▌")
                # final update without cursor
                placeholder_e.markdown(exercises_acc)
                exercises = exercises_acc

if __name__ == "__main__":
    main()