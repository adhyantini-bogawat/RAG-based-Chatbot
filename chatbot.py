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
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate, metrics, RunConfig
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall, \
summarization_score, faithfulness
expected_contexts = ["Increase, motivation"]
retrieved_contexts = ["Increase, motivation, wake-up time, scheduling, preparing"]

def calculate_context_metrics(expected_contexts, retrieved_contexts):
    # Assume expected_contexts and retrieved_contexts are lists of labels
    precision = precision_score(expected_contexts, retrieved_contexts, average='micro')
    recall = recall_score(expected_contexts, retrieved_contexts, average='micro')
    relevance = np.mean([1 if expected in retrieved_contexts else 0 for expected in expected_contexts])
    return {
    "context_precision": precision,
    "context_recall": recall,
    "context_relevance": relevance
    }

context_metrics = calculate_context_metrics(expected_contexts, retrieved_contexts)

def calculate_generation_metrics(expected_answers, generated_answers):
    faithfulness = np.mean(
    [1 if generated == expected else 0 for generated, expected in zip(generated_answers, expected_answers)])
    answer_relevance = np.mean(
    [1 if word in generated else 0 for generated, expected in zip(generated_answers, expected_answers) for word in
    expected.split()])
    return {
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevance
    }

expected_answers = [
"1. Set a consistent wake-up time each day to establish a regular routine. 2. Prepare as much as possible the night before, such as laying out clothes, preparing breakfast, or planning your day's schedule.3. Create a pleasant morning environment by setting up a comfortable and inviting space. This could include playing soft music, using aromatherapy, or opening the curtains to let in natural light.4. Start your day with an activity that energizes you and triggers positive feelings, like exercise, journaling, or meditation. These habits can help set a productive tone for the rest of the day.5. Set clear goals for the day, focusing on the outcomes you want rather than the specific tasks. This helps maintain motivation by keeping your focus on the end result.6. Gradually increase the difficulty level of your morning routine to build resilience and overcome challenges that may arise during the day.7. If possible, avoid checking emails or social media first thing in the morning, as these can be distractions that drain energy and motivation.8. Celebrate small successes throughout the day to reinforce positive habits and maintain a sense of accomplishment."
]
# Fetch the document
pdfloader = PyPDFLoader("D:/NEU/PromptEng/Adhyantini_Bogawat_002766612/Atomic_habits.pdf")
pdfpages = pdfloader.load_and_split()
# Loading the document using Langchains inbuilt extractors, formatters, loaders, embeddings and LLM's
mydocuments = pdfloader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(mydocuments)
# Load the model
model = Ollama(model="mistral")
# Use embedding model to convert texts into embeddings and store in the vector database
# vectorstore = Chroma.from_documents(documents = texts, collection_name = "rag_chroma", embedding =OllamaEmbeddings(model='nomic-embed-text'))
# Another vector database called FAISS
vectorstore = FAISS.from_documents(texts, embedding=OllamaEmbeddings(model='nomic-embed-text'))
retriever = vectorstore.as_retriever()

def process_input(question):
    # perform the RAG
    after_rag_template = """ Answer the question based only on the following context, suggest top 3 recommendations, and ask a follow-up question to keep the conversation going:{context} Question: {question} """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = ({"context": retriever,
    "question": RunnablePassthrough()} | after_rag_prompt | model | StrOutputParser())
    answer = after_rag_chain.invoke(question)
    context_texts = [doc.page_content for doc in texts]
    # Calculate metrics (example with dummy data)
    context_metrics = calculate_context_metrics(context_texts, context_texts)
    generation_metrics = calculate_generation_metrics(expected_answers, [answer])
    return answer, context_metrics, generation_metrics, context_texts

st.title("A guide to help you become more motivated!")
st.write("Ask the chatbot on tips to feel more motivated")
# Input fields
question = st.text_input("Question")
# Button to process input
if st.button('Enter'):
    with st.spinner('Processing...'):
        answer, context_metrics, generation_metrics, context_texts = process_input(question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
        st.write("Context Metrics:", context_metrics)
        st.write("Generation Metrics:", generation_metrics)
        queries = ["How can I be more motivated in the mornings?"]
        expected_answers = [
        "1. Set a consistent wake-up time each day to establish a regular routine. 2. Prepare as much as possible the night before, such as laying out clothes, preparing breakfast, or planning your day's schedule.3. Create a pleasant morning environment by setting up a comfortable and inviting space. This could include playing soft music, using aromatherapy, or opening the curtains to let in natural light.4. Start your day with an activity that energizes you and triggers positive feelings, like exercise, journaling, or meditation. These habits can help set a productive tone for the rest of the day.5. Set clear goals for the day, focusing on the outcomes you want rather than the specific tasks. This helps maintain motivation by keeping your focus on the end result.6. Gradually increase the difficulty level of your morning routine to build resilience and overcome challenges that may arise during the day.7. If possible, avoid checking emails or social media first thing in the morning, as these can be distractions that drain energy and motivation.8. Celebrate small successes throughout the day to reinforce positive habits and maintain a sense of accomplishment."
        ]
        results = []
        for query, expected in zip(queries, expected_answers):
            results.append({
            "question": query,
            "ground_truth": expected,
            "answer": answer,
            "contexts": context_texts # Add the extracted context texts here
            })
        results_df = pd.DataFrame(results)
        results_ds = Dataset.from_pandas(results_df)
        # metrics=[answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall,
        # faithfulness],
        # run_config = RunConfig(timeout=300, max_wait= 300) # Set timeout to 180 seconds (3 minutes)
        run_config = RunConfig(timeout=600)
        evaluation_report = evaluate(
        results_ds,
        metrics=[answer_similarity],
        llm=model,
        embeddings=OllamaEmbeddings(model='nomic-embed-text'),
        run_config=run_config
        )
        evaluation_report_df = evaluation_report.to_pandas()
        st.write("Evaluation Report:", evaluation_report_df)