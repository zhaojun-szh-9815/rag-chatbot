from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import pinecone
from dotenv import load_dotenv
import os
import time
from enum import Enum, auto
import time

class pinecone_operation(Enum):
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    READ = auto()

    def get_operation(s: str):
        if s == "create":
            return pinecone_operation.CREATE
        elif s == "update":
            return pinecone_operation.UPDATE
        elif s == "delete":
            return pinecone_operation.DELETE
        else:
            return pinecone_operation.READ

def environment_setup():
    load_dotenv()
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT')
    )

def model_setup(selection: str) -> dict:
    if selection == "openai":
        return {
            "embedding": OpenAIEmbeddings(model="text-embedding-ada-002"),
            "dimension": 1536,
            "metric": "cosine",
            "chat": ChatOpenAI(
                        openai_api_key=os.environ["OPENAI_API_KEY"],
                        model='gpt-3.5-turbo'
                    )
        }
    elif selection == "huggingface":
        return {
            "embedding": HuggingFaceInstructEmbeddings(model_name="intfloat/e5-small-v2"),
            "dimension": 384,
            "metric": "cosine",
            "chat": HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.7, "max_length":512})
        }
    else:
        return {}

def pinecone_operate(operation: pinecone_operation, model: dict) -> Pinecone|None:
    index_name = os.environ["PINECONE_INDEX_NAME"]
    if operation == pinecone_operation.CREATE:
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
        pinecone.create_index(
            index_name,
            dimension=model["dimension"],
            metric=model["metric"]
        )
        # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
        return pinecone.Index(index_name)
    elif operation == pinecone_operation.UPDATE or operation == pinecone_operation.READ:
        if index_name not in pinecone.list_indexes():
            print("index not found")
        return pinecone.Index(index_name)
    else:
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
        return

def load_file(filename: str, model: dict) -> Pinecone:
    index_name = os.environ["PINECONE_INDEX_NAME"]
    raw_documents = Docx2txtLoader(filename).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    documents = text_splitter.split_documents(raw_documents)
    return Pinecone.from_documents(documents, model["embedding"], index_name=index_name)

def augment_prompt(query: str, chat_history: list, pinecone_connection: Pinecone):
    results = pinecone_connection.similarity_search(query, k=3)
    print(results)
    source_knowledge = "\n".join([x.page_content for x in results])
    history = "\n".join([x.content for x in chat_history])
    augmented_prompt = f"""Based on the contexts below and the chat history, conclude the contexts and answer the question in your own words. You can answer you don't know if you cannot find related information from contexts.

    Contexts:
    {source_knowledge}.

    Chat history:
    {history}.

    Question: {query}"""
    return augmented_prompt

def chat_llm(question: str, messages: list, chat: HuggingFaceHub, pinecone_connection: Pinecone):
    if len(messages) > 4:
        messages = messages[0] + messages[-4:]
        messages = messages.messages
    prompt = augment_prompt(query=question, chat_history=messages, pinecone_connection=pinecone_connection)
    messages.append(HumanMessage(content=question))

    # print(prompt)

    res = chat(prompt)
    messages.append(AIMessage(content=res))

    return res

def client_chat_setup():
    environment_setup()
    model = model_setup(selection="huggingface")
    docsearch = Pinecone.from_existing_index(os.environ["PINECONE_INDEX_NAME"], model["embedding"])
    messages = [
        SystemMessage(content="You are a experience agent helping candidates seeking jobs. Based on the candidates' information, answer the questions from the employer with your own words."),
    ]
    return model, docsearch, messages

def admin_vectordb_management():
    environment_setup()
    model = model_setup(selection="huggingface")

    operation = pinecone_operation.get_operation(input("operation:\n"))
    vectorstore = pinecone_operate(operation=operation, model=model)
    print(vectorstore.describe_index_stats())
    previous_count = vectorstore.describe_index_stats()["total_vector_count"]

    if operation == pinecone_operation.CREATE or operation == pinecone_operation.UPDATE:
        docsearch = load_file(filename="server/introduction.docx", model=model)

        while vectorstore.describe_index_stats()["total_vector_count"] == previous_count:
            print("processing...")
            time.sleep(2)
    else:
        docsearch = Pinecone.from_existing_index(os.environ["PINECONE_INDEX_NAME"], model["embedding"])

    messages = [
        SystemMessage(content="You are a helpful assistant. You are helping a person seeking the job. Image you are the candidate and ready to answer questions from the interviewer with the person's information."),
    ]

    while True:
        question = input("question:\n")
        if question == "exit":
            break
        response = chat_llm(question=question, messages=messages, chat=model["chat"], pinecone_connection=docsearch)
        print(response)


if __name__ == "__main__":
    admin_vectordb_management()