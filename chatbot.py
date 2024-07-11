import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# Fetch the document
pdfloader = PyPDFLoader("/path/to/your/file")
pdfpages = pdfloader.load_and_split()

def process_input(question):
    # Loading the document using Langchains inbuilt extractors, formatters, loaders, embeddings and LLM's
    mydocuments = pdfloader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(mydocuments)

    #Load the model
    model = Ollama(model="mistral")

    #Use embedding model to convert texts into embeddings and store in the vector database
    # vectorstore = Chroma.from_documents(documents = texts, collection_name = "rag_chroma", embedding =OllamaEmbeddings(model='nomic-embed-text'))
    
    # Another vector database called FAISS
    vectorstore = FAISS.from_documents(texts, embedding =OllamaEmbeddings(model='nomic-embed-text'))
    retriever = vectorstore.as_retriever()


    #perform the RAG 

    after_rag_template = """ Answer the question based only on the following context, suggest top 3 recommendations, and ask a follow-up question to keep the conversation going:{context} Question: {question} """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | after_rag_prompt | model | StrOutputParser())

    return after_rag_chain.invoke(question)

st.title("A guide to help you become more motivated !")
st.write("Ask the chatbot on tips to feel more motivated")

# Input fields
question = st.text_input("Question")

# Button to process input
if st.button('Enter'):
    with st.spinner('Processing...'):
        answer = process_input( question)
        st.text_area("Answer", value=answer, height=300, disabled=True)




    


