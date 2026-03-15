from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint

file_path = "./attention_is_all_you_need.pdf"
loader = PyPDFLoader(file_path) 
doc = loader.load()

print(doc)
print(doc[0].metadata)
print(doc[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

all_splits = text_splitter.split_documents(doc)

print(f"Split blog post into {len(all_splits)} sub-documents.")
print(f"Metadata: {all_splits[0].metadata}")
