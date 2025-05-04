import streamlit as st
from tools_api import *
import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# Hard‑coded channel we want to show on startup
CHANNEL_ID = "UCch2JvY2ZSwcjf5gb93HGQw"

def get_top_videos(channel_id: str, max_results: int = 5):
    """
    Return a list with the `max_results` most‑viewed videos from the given
    YouTube channel.  Each element is a dict: {"title": str, "video_id": str}.
    """
    url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?key={YOUTUBE_API_KEY}"
        f"&channelId={channel_id}"
        f"&part=snippet"
        f"&order=viewCount"
        f"&maxResults={max_results}"
        f"&type=video"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    videos = [
        {
            "title": item["snippet"]["title"],
            "video_id": item["id"]["videoId"],
            "thumbnail": (
                item["snippet"]["thumbnails"].get("high") or
                item["snippet"]["thumbnails"].get("medium") or
                item["snippet"]["thumbnails"]["default"]
            )["url"],
        }
        for item in data.get("items", [])
    ]
    return videos

def main():
    st.set_page_config(layout="wide")
    # ---------------- Sidebar: per‑user OpenAI key ----------------
    with st.sidebar:
        st.header("Configuration")
        if "user_openai_api_key" not in st.session_state:
            openai_key = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                placeholder="sk‑..."
            )
            if st.button("Save Key"):
                if openai_key:
                    st.session_state["user_openai_api_key"] = openai_key
                    # Make the key available to OpenAI client inside this session only
                    os.environ["OPENAI_API_KEY"] = openai_key
                    st.success("Key saved for this session.")
                    st.experimental_rerun()
        else:
            st.success("OpenAI key saved for this session.")

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
    # Require per-user key before continuing
    # ------------------------------------------------------------------
    if "user_openai_api_key" not in st.session_state:
        st.info("Please enter your OpenAI API key in the sidebar to start.")
        st.stop()

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