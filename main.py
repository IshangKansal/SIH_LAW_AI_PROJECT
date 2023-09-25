from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

offload_folder = r"C:\Users\ishan\.cache\huggingface\hub"

loader = DirectoryLoader("ipc-data", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

texts = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

db = Chroma.from_documents(texts, embeddings)

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=offload_folder)

model = TFAutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, from_pt=True, cache_dir=offload_folder
)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.95,
    device=0,
)

local_llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)


def code(input_query):
    llm_response = qa_chain({"query": input_query})
    return llm_response["result"]


# input_query = str(input("Enter: "))
# print(code(input_query))
