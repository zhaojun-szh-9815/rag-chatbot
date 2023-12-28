# RAG Chat Flask Backend

app.py: It is a Flask backend service with two APIs.
- '/' is to setup hugggingface/openai and pinecone connections
- '/chat' is to execute LLM service according to the message in POST

---

chatService.py: It is RAG Chatbot service.
- Using LLM model 'google/flan-t5-xxl' and embedding model 'intfloat/e5-small-v2' from HuggingFace
- Option: LLM model ChatGPT and embedding model 'text-embedding-ada-002' from OpenAI (will get better result)
- Use vectordatabase Pinecone to ensure persistence, avoid extract documents every time
- Customize prompt to allow LLM refer the knowledges from vector database and the chat history

---

please check requirements.txt for necessary libraries

start backend by `flask run`
