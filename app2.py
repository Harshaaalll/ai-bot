from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import pickle
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import os

def download_model(url, destination_path):
    # Stream the file from the URL in chunks
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Example URL where the model file is hosted
model_url = "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf?download=true"

# Destination path where the model file will be saved
model_path = "stablelm-zephyr-3b.Q4_K_M.gguf"

# Download the model file
download_model(model_url, model_path)
app = Flask(__name__)
CORS(app)


def prepare_docs(pdf):
    docs = []
    metadata = []
    content = []

    # for pdf in pdf_docs:

    pdf_reader = PyPDF2.PdfReader(pdf)
    for index, text in enumerate(pdf_reader.pages):
        doc_page = {'title': 'title' + " page " + str(index + 1),
                    'content': pdf_reader.pages[index].extract_text()}
        docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    print("Content and metadata are extracted from the documents")
    return content, metadata

def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db
template = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
def get_conversation_chain(vectordb):
    llama_llm = LlamaCpp(
    model_path="stablelm-zephyr-3b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=200,
    top_p=1,
    callback_manager=callback_manager,
    n_ctx=3000)

    retriever = vectordb.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llama_llm,
                           retriever=retriever,
                           #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain


@app.route('/upload-document', methods=['POST'])
def upload_document():
    file = request.files['file']
    docs = prepare_docs(file)
    split_docs = get_text_chunks(docs[0], docs[1])
    vectordb = ingest_into_vectordb(split_docs)
    conversation_chain = get_conversation_chain(vectordb)
    with open('conversation_chain.pkl', 'wb') as f:
        pickle.dump(conversation_chain, f)
    return jsonify({'message': 'Document processed successfully'})

@app.route('/ask-question', methods=['POST'])
def ask_question():
    if request.is_json:
        user_question = request.json.get('question')
        if user_question:
            try:
                with open('conversation_chain.pkl', 'rb') as f:
                    conversation_chain = pickle.load(f)
                response = conversation_chain({"question": user_question})
                return jsonify({'answer': response['answer']})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'No question provided'}), 400
    else:
        return jsonify({'error': 'Request content-type must be application/json'}), 415


if __name__ == '__main__':
    app.run(debug=True)
