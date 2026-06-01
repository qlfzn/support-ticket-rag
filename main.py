import os
import dotenv
import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

dotenv.load_dotenv()
CHROMA_PATH = "chroma"


def load_document(path: str) -> list[str]:
    reader = PdfReader(path)

    document = []
    for page in reader.pages:
        document.append(page.extract_text())

    return document


def load_tickets_data(path: str):
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


def chunk_document_text(document: list[str], chunk_size: int):
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


doc = load_document("./data/techflow_rule_docs.pdf")
ticket_chunks = load_tickets_data("./data/customer_support_tickets_200k.csv")
print("\nChunking text from PDF...")
pdf_chunks = chunk_document_text(doc, 1000)

print("\n", pdf_chunks[0:3])
print("\n", ticket_chunks[3])

doc_collection = create_embeddings(pdf_chunks, ticket_chunks)
coll = chromadb.PersistentClient("./chroma").get_collection("tech-policy-tickets")

# if coll:
#     results = coll.query(
#         query_texts=["What is the refund window?"],
#         n_results=1,
#         where={"source": "policy_doc"}
#     )
#     print(results["documents"])