from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

class YouTubeTranscriptProcessor:
    """Handles loading and splitting YouTube video transcripts."""

    def __init__(self, url):
        self.url = url

    def load_and_split(self):
        try:
            loader = YoutubeLoader.from_youtube_url(self.url)
            transcript = loader.load()
            text = transcript[0].page_content
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.create_documents([text])
        except Exception as e:
            raise RuntimeError(f"Error loading/splitting transcript: {e}")


class VectorDatabase:
    """Handles embedding and vector store creation."""

    def __init__(self, docs, persist_dir="chroma_youtube_db"):
        self.docs = docs
        self.persist_dir = persist_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        self.vectorstore = None

    def create_vectorstore(self):
        try:
            self.vectorstore = Chroma.from_documents(
                documents=self.docs,
                embedding=self.embedding_model,
                persist_directory=self.persist_dir
            )
        except Exception as e:
            raise RuntimeError(f"Error creating vectorstore: {e}")

    def retrieve(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)


class LLMResponder:
    """Handles prompt creation and calling the LLM."""

    def __init__(self, api_key):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self):
        return PromptTemplate.from_template("""
You are a helpful assistant specialized in answering questions about YouTube videos based solely on their transcripts.

Use only the factual information provided in the transcript below to answer the user's question:

Transcript:
{docs}

Question:
{question}

— Do not make assumptions or add information that is not explicitly stated in the transcript.
— If the transcript does not provide enough information to answer the question, respond with:
"I don't know."

Your answer should be detailed, clear, and verbose.
""")

    def answer(self, retrieved_docs, question):
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = self.prompt_template.format(docs=retrieved_text, question=question)
        return self.llm.invoke(prompt)


class RAGApplication:
    """Main app flow."""

    def __init__(self, youtube_url, api_key):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.youtube_url = youtube_url
        self.api_key = api_key
        self.vector_db = None
        self.responder = None

    def setup(self):
        print("🔄 Processing YouTube transcript...")
        processor = YouTubeTranscriptProcessor(self.youtube_url)
        docs = processor.load_and_split()

        print("💾 Creating vector database...")
        self.vector_db = VectorDatabase(docs)
        self.vector_db.create_vectorstore()

        print("⚙️ Initializing LLM...")
        self.responder = LLMResponder(self.api_key)

    def run(self):
        print("\n💬 Ask anything about the video transcript (type 'exit' to quit):")
        while True:
            query = input("\n❓ Your question: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting. Thanks for using the RAG app!")
                break

            print("🔍 Retrieving info...")
            retrieved_docs = self.vector_db.retrieve(query)

            print("🤖 Getting LLM response...\n")
            response = self.responder.answer(retrieved_docs, query)

            print(f"📢 Answer:\n{response}")


# --- RUN APP ---
if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env

    YOUTUBE_URL = os.getenv("YOUTUBE_URL")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    app = RAGApplication(YOUTUBE_URL, GOOGLE_API_KEY)
    app.setup()
    app.run()