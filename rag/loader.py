from langchain_community.document_loaders import PyPDFLoader
import os


def load_documents(folder_path="data/documents"):

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            file_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            for doc in docs:

                # remove empty pages
                if doc.page_content.strip():

                    # clean extra whitespace
                    doc.page_content = doc.page_content.replace("\n", " ").strip()

                    documents.append(doc)

    return documents