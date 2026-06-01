import os
import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(path: str) -> list[str]:
    reader = PdfReader(path)
    
    document = []
    for page in reader.pages:
        document.append(page.extract_text())
    
    return document

def load_tickets_data(path: str):
    tickets_df = pd.read_csv(path)

    # create column for embedding
    tickets_df['combined_text'] = tickets_df['issue_description'] + " | " + tickets_df['category'] + " | " + tickets_df['resolution_notes']

    train, test = train_test_split(
        tickets_df,
        test_size=0.2,
        stratify=tickets_df['category'],
        random_state=42
    )

    if "test_tickets_data.csv" not in os.listdir("./data/"):
        print("Creating test dataset")
        test_df = pd.DataFrame(test)
        test_df.to_csv('./data/test_tickets_data.csv', sep=",", index=False)

    return train, test

def chunk_document_text(document: list[str], chunk_size: int):
    # chunking text to create smaller text size
    docs = [Document(page_content=text) for text in document]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n"],
        chunk_size=chunk_size,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} document into {len(chunks)} chunks.")
    
    return chunks

def create_embeddings(text_chunks):
    return


doc = load_document("./data/techflow_rule_docs.pdf")
train, test = load_tickets_data("./data/customer_support_tickets_200k.csv")
print("\nChunking text from PDF...")
pdf_chunks = chunk_document_text(doc, 1000)
print("\nChunking text from CSV tickets data...")
ticket_chunks = chunk_document_text(train, 1000)