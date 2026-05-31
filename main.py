import os
import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split


def load_document(path) -> dict[int, str]:
    reader = PdfReader(path)
    
    document = {}
    num = 1 
    for page in reader.pages:
        document[num] = page.extract_text()
        num += 1
    
    return document

def load_tickets_data(path):
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

    return tickets_df, train, test

def chunk_document_text(pages, chunk_size, overlap):
    # chunking text to create smaller text size
    # start with size 1024


    return


doc = load_document("./data/techflow_rule_docs.pdf")
df, train, test = load_tickets_data("./data/customer_support_tickets_200k.csv")
