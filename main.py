import os
import dotenv
import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from groq import Groq

dotenv.load_dotenv()
CHROMA_PATH = "./chroma"


def load_document(path: str) -> list[str]:
    reader = PdfReader(path)

    document = []
    for page in reader.pages:
        document.append(page.extract_text())

    return document


def load_tickets_data(path: str) -> list[Document]:
    tickets_df = pd.read_csv(path)

    # create column for embedding
    tickets_df["combined_text"] = (
        tickets_df["issue_description"] + " | " + tickets_df["resolution_notes"]
    )

    train, test = train_test_split(
        tickets_df, test_size=0.2, stratify=tickets_df["category"], random_state=42
    )

    if "test_tickets_data.csv" not in os.listdir("./data/"):
        print("Creating test dataset")
        test_df = pd.DataFrame(test)
        test_df.to_csv("./data/test_tickets_data.csv", sep=",", index=False)

    ticket_chunks = [
        Document(
            metadata={
                "source": "tickets",
                "ticket_id": ticket_id,
                "category": row["category"],
            },
            page_content=row["combined_text"],
        )
        for ticket_id, row in train.iterrows()
    ]

    return ticket_chunks


def chunk_document_text(document: list[str], chunk_size: int) -> list[Document]:
    # chunking text to create smaller text size
    docs = [
        Document(metadata={"source": "policy_doc"}, page_content=text)
        for text in document
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["^\d+\.\s+[A-Z][A-Z\s]+$"],
        chunk_size=chunk_size,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} document into {len(chunks)} chunks.")

    return chunks


def create_embeddings(text_chunks: list[Document], ticket_chunks: list[Document]):
    if os.path.isdir(CHROMA_PATH):
        print("\nChroma path already exists. Skipping...")
    
    client = chromadb.PersistentClient("./chroma")

    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name="tech-policy-tickets",
        embedding_function=embed_func,
    )

    all_chunks = text_chunks + ticket_chunks
    batch_size = 5000

    for i in range(0, len(all_chunks), batch_size):
        end = i + batch_size
        print(f"Current batch range: {i} - {end}")
        collection.add(
            documents=[chunk.page_content for chunk in all_chunks[i:end]],
            metadatas=[chunk.metadata for chunk in all_chunks[i:end]],
            ids=[str(j) for j in range(len(all_chunks[i:end]))],
        )

    return collection

def query_db(query: str, source: str) -> chromadb.QueryResult:
    coll = chromadb.PersistentClient("./chroma").get_collection("tech-policy-tickets")

    results = coll.query(
        query_texts=[query],
        n_results=2,
        where={"source": source}
    )

    return results

def generate_response(policy, tickets, new_ticket):
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
        Based on these guidelines:
        {policy}

        Here's a new support ticket:
        {new_ticket}

        Suggest a response that:
        1. Follows the policy
        2. Is professional and empathetic
        3. Cites the relevant policy or past precedent
        4. Provides clear next steps
    """

    text = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a support agent assistant."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return text.choices[0].message.content

# doc = load_document("./data/techflow_rule_docs.pdf")
# ticket_chunks = load_tickets_data("./data/customer_support_tickets_200k.csv")
# print("\nChunking text from PDF...")
# pdf_chunks = chunk_document_text(doc, 1000)

# if os.path.isdir(CHROMA_PATH):
#     print("\nChroma path already exists. Skipping...")
# else:
#     doc_collection = create_embeddings(pdf_chunks, ticket_chunks)

policy_retrieved = query_db(
    query="I would like to request a refund for the recent charge.",
    source="policy_doc"
)

tickets_retrieved = query_db(
    query="I would like to request a refund for the recent charge.",
    source="tickets"
)

text = generate_response(policy_retrieved, tickets_retrieved, "The payment was deducted from my bank account but the transaction shows failed.")
print("\n",text)