from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os  
from langchain_openai import ChatOpenAI, OpenAI  # switched to OpenAI models
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from tqdm import tqdm
from dotenv import load_dotenv

import streamlit as st
from typing import List


# Load environment variables
load_dotenv()


def parse_url(url: str) -> str:
    """
    Extract video ID from URL.

    Args: 
        url(str): youtube video url

    Returns:
        Youtube video's video ID
    
    """
    if "youtu.be" in url:
        return url.split("/")[-1]
    if "=" in url:
        return url.split("=")[-1]

    return url

def get_text_from_video(url: str) -> str:
    """
    Get transcript text from YouTube video.

    Args:
        url(str): youtube video url
    """
    video_id = parse_url(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['nl', 'en'])
    return [entry['text'] for entry in transcript]

def create_chunks(transcript_text: str) -> list:
    """
    Split transcript text into processable chunks.

    Args:
        transcript_text (str): Youtube video's transcripted text

    Returns:
        processable chunks
    
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(" ".join(transcript_text))
    return chunks


# ---------------------------------------------------------------------------
# Line‑by‑line streaming version – expects `transcript_lines` as a list[str]
# ---------------------------------------------------------------------------
def english_translate_stream_lines(transcript_lines: List[str]):
    """
    Stream an English translation for a *list* of Dutch text segments
    (e.g. words or short sentences). Each list item is translated in turn;
    after finishing an item we yield a newline, so the caller sees one
    English line per Dutch line.
    """
    chat_llm = ChatOpenAI(streaming=True, model="gpt-4o-mini", temperature=0)

    system_msg = (
        "You are a professional Dutch‑to‑English translator. "
        "Translate any Dutch text in the same order as it appears. "
        "NEVER add comments like 'Here is the translation'. "
        "Output *only* the pure English translation."
    )

    partial = ""
    for idx, dutch_segment in enumerate(transcript_lines):
        # Build a minimal message context for this segment
        messages = [
            ("system", system_msg),
            ("user", f"Translate the following Dutch text to English:\n\n{dutch_segment}"),
        ]

        first_real_token_seen = False
        buffer_this_segment = ""

        for msg_piece in chat_llm.stream(messages):
            token = getattr(msg_piece, "content", "")
            if not token:
                continue
            # prevent unintended line‑breaks inside a single segment
            token = token.replace("\n", " ").replace("\r", " ")

            # Strip meta prefaces on the first token
            if not first_real_token_seen:
                lowered = token.lower()
                if lowered.startswith(("here is", "here's", "translation", "the translation")):
                    continue
                first_real_token_seen = True

            buffer_this_segment += token
            partial += token
            yield partial  # send each incremental token

        # Finished this list item – append a markdown line break and yield
        partial += "  \n"
        yield partial


# ---------------------------------------------------------------------------
# Streaming helpers for vocab list, grammar points, and exercises
# ---------------------------------------------------------------------------

def get_vocab_stream(chunks: List[str]):
    """
    Same logic as get_vocab(), but streams the **final** consolidation step so
    callers (Streamlit) can show a typing effect.  The per‑chunk extraction
    remains synchronous; only the final LLM pass is streamed.
    """
    # 1 · non‑stream: extract vocab items from each chunk
    llm_sync = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # chat completions endpoint
    template = (
        "Text: {text}\n"
        "Goal: You are a Dutch language expert. Extract a list of essential Dutch "
        "vocabulary used in the given text. Avoid extracting people names or place names."
    )
    prompt = ChatPromptTemplate.from_template(template)
    # original for‑loop kept for clarity
    with st.spinner("Extracting vocabulary..."):
        vocab_list = [ (prompt | llm_sync).invoke({"text": chunk}).content.strip() for chunk in chunks ]
        combined_vocab = " ".join(vocab_list)

    # 2 · stream: consolidate & translate vocab list
    chat_llm = ChatOpenAI(streaming=True, model="gpt-4o-mini", temperature=0)
    sys_msg = (
        "You are a Dutch language expert. For the given vocabulary list, "
        "create a coherent single vocabulary list. "
        "add the English translation for each word."
        "Avoid repeating words and remove duplicates."
        "Avoid using the word 'list' in the answer."
        "NEVER add any introductory sentences or comments like 'OK, Here are the exercises'."
        "NEVER ask follow-up questions and provide the requested information only."
    )
    messages = [
        ("system", sys_msg),
        ("user", combined_vocab),
    ]

    partial = ""
    for msg_piece in chat_llm.stream(messages):
        token = getattr(msg_piece, "content", "")
        if token:
            partial += token
            yield partial


def extract_grammar_stream(chunks: List[str]):
    """
    Stream the *explanation* phase of extract_grammar(); the initial extraction
    per chunk stays synchronous.
    """
    # 1 · non‑stream: gather grammar tokens per chunk
    llm_sync = ChatOpenAI(model="gpt-4o", temperature=0.2)  # chat completions endpoint
    template = (
        "Text: {text}\n"
        "Goal: You are a Dutch grammar expert. Extract the key grammar points from "
        "the given text. Provide an example from the text for each grammar point. "
        "Answer: List the key grammar points separated by commas."
    )
    prompt = ChatPromptTemplate.from_template(template)
    with st.spinner("Extracting grammar points..."):
        grammar_list = [ (prompt | llm_sync).invoke({"text": chunk}).content.strip() for chunk in chunks ]

        all_grammar = set()
        for grammar in grammar_list:
            all_grammar.update(t.strip() for t in grammar.split(",") if t.strip())

        grammar_points = ", ".join(sorted(all_grammar))

    # 2 · stream: explain grammar points
    chat_llm = ChatOpenAI(streaming=True, model="gpt-4o", temperature=0)
    sys_msg = (
        "You are a Dutch grammar expert."
        "Provide a brief explanation for each grammar point and preserve the example from the text."
        "do not omit or repeat anything."
        "NEVER ask any follow-up questions"
        "NEVER add any introductory sentences or comments like 'OK, Here are the' or 'Ok, Here's the'."
        "**NEVER** add any questions starting with 'Do you want me to' or 'Would you like me to' at the end of your response."
    )
    messages = [
        ("system", sys_msg),
        ("user", grammar_points),
    ]

    partial = ""
    for msg_piece in chat_llm.stream(messages):
        token = getattr(msg_piece, "content", "")
        if token:
            partial += token
            yield partial


def create_exercises_stream(vocab_list: str, grammar_points: str):
    """
    Stream generation of exercises and answers based on vocab + grammar.
    """
    # num_predict tells Ollama how many tokens it may emit; raise it so the
    # stream does not cut off after the first exercise block.
    chat_llm = ChatOpenAI(
        streaming=True,
        model="gpt-4o",
        temperature=0,
        max_tokens=2048
    )
    sys_msg = (
        "You are a Dutch language teacher. "
        "Create a set of exercises to help students practice the given vocabulary and grammar points. "
        "Include fill-in-the-blank, sentence construction, and multiple-choice questions."
        "provide the answers for each exercise at the end."
        "NEVER ask any follow-up questions and *ONLY* provide the exercises and answers."
        "NEVER add any introductory sentences or comments like 'OK, Here are the exercises'."
    )
    user_msg = (
        f"Vocabulary:\n{vocab_list}\n\nGrammar Points:\n{grammar_points}\n\n"
        "Include fill-in-the-blank, sentence construction, and multiple-choice questions."
        "provide the answers for each exercise at the end."
        "NEVER ask any follow-up questions and *ONLY* provide the exercises and answers."
        "Please produce the exercises now."
    )

    messages = [
        ("system", sys_msg),
        ("user", user_msg),
    ]

    partial = ""
    for msg_piece in chat_llm.stream(messages):
        token = getattr(msg_piece, "content", "")
        if token:
            partial += token
            yield partial
    # ensure the last token flushes any buffered markdown list
    partial += "  \n"
    yield partial



def process_user_query(query, chunks, url):
    """Select appropriate tool based on user query and generate response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # chat model for the agent
    
    # Create wrapper functions for tools
    def get_vocab_wrapper(input_str=""):
        return get_vocab_stream(chunks)
    
    def extract_grammar_wrapper(input_str=""):
        return extract_grammar_stream(chunks)
    
    def english_translate_wrapper(input_str=""):
        return english_translate_stream_lines(" ".join(chunks))
    
    def create_exercises_wrapper(input_str=""):
        vocab_list = get_vocab_stream(chunks)
        grammar_points = extract_grammar_stream(chunks)
        return create_exercises_stream(vocab_list, grammar_points)
    
    # Create tools with wrapper functions
    tools = [
        Tool(
            name="get_vocab",
            func=get_vocab_wrapper,
            description="Extracts a list of Dutch vocabulary from the transcript."
        ),
        Tool(
            name="extract_grammar",
            func=extract_grammar_wrapper,
            description="Extracts main grammar points from the transcript."
        ),
        Tool(
            name="english_translate",
            func=english_translate_wrapper,
            description="Provides a sentence-by-sentence English translation of the transcript."
        ),
        Tool(
            name="create_exercises",
            func=create_exercises_wrapper,
            description="Creates Dutch language exercises based on vocabulary and grammar points."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Call agent with user query
    response = agent.invoke(input=query, handle_parsing_errors=True)

    return response