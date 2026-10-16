# YT_AI_CHATBOT

# ==========================================
# YOUTUBE AI STUDY ASSISTANT USING RAG
# ==========================================

!pip -q install -U langchain-google-genai langchain-community langchain-text-splitters youtube-transcript-api faiss-cpu

import os
import re
from google.colab import userdata
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. API KEY AND AI MODELS
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=2,
    timeout=45
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 2. EXTRACT YOUTUBE VIDEO ID
def get_video_id(url):
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL.")


# 3. FETCH VIDEO TRANSCRIPT
def get_transcript(video_id):
    api = YouTubeTranscriptApi()

    for language in ["en", "hi"]:
        try:
            result = api.fetch(
                video_id,
                languages=[language]
            )

            transcript = " ".join(
                item.text for item in result
                if item.text.strip()
            )

            if transcript:
                return transcript

        except Exception:
            continue

    raise ValueError(
        "No accessible English or Hindi transcript found."
    )


# 4. SPLIT TRANSCRIPT INTO CHUNKS
def create_chunks(transcript):
    document = Document(page_content=transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250
    )

    return splitter.split_documents([document])


# 5. AUTOMATICALLY GENERATE QUESTIONS
def generate_questions(transcript):
    # Sample different sections if the video is very long.
    if len(transcript) > 18000:
        part = 6000
        middle = len(transcript) // 2

        text = (
            transcript[:part]
            + "\n...\n"
            + transcript[middle-part//2:middle+part//2]
            + "\n...\n"
            + transcript[-part:]
        )
    else:
        text = transcript

    prompt = f"""
You are an educational assistant.

Read the video transcript and generate 8 useful questions
that can be answered from it.

Include basic concepts, explanations, processes, and details.
Do not provide the answers.
Do not invent information outside the transcript.

Return a numbered list from 1 to 8.

TRANSCRIPT:
{text}
"""

    response = llm.invoke(prompt)
    return response.content


# 6. ANSWER QUESTIONS USING VIDEO CONTEXT
def answer_question(vector_store, question):
    # Retrieve the 3 most relevant transcript chunks.
    documents = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = f"""
You are a YouTube video question-answering assistant.

Answer the question using ONLY the transcript excerpts below.
Do not use outside knowledge or invent facts.

If the answer is not available, respond:
"I couldn't find that information in the available video transcript."

Keep the answer clear and useful.

TRANSCRIPT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content


# 7. MAIN PROGRAM
url = input("Paste your YouTube video link: ").strip()

try:
    video_id = get_video_id(url)

    print("\nFetching transcript...")
    transcript = get_transcript(video_id)
    print("Transcript fetched:", len(transcript), "characters")

    print("\nCreating chunks...")
    chunks = create_chunks(transcript)
    print("Number of chunks:", len(chunks))

    print("\nCreating FAISS vector database...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print("Video indexed successfully!")

    print("\nGenerating questions from the video...")
    questions = generate_questions(transcript)

    print("\n========== VIDEO-BASED QUESTIONS ==========")
    print(questions)

    print("\n========== ASK YOUR OWN QUESTIONS ==========")
    print("Ask anything about the video.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "stop"]:
            print("Study session ended.")
            break

        if not question:
            continue

        try:
            answer = answer_question(
                vector_store,
                question
            )
            print("\nAI Answer:", answer)

        except Exception as error:
            print("\nAnswer request failed:", error)
            print("Try again after a short wait.")

except Exception as error:
    print("\nCould not process the video:", error)