from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
import chatService
import traceback 

app = Flask(__name__, static_folder='../client/build', static_url_path='')
cors = CORS(app)
model = None
docsearch = None
messages = None

@app.route('/setup', methods = ['GET'])
@cross_origin()
def setup():
    global model
    global docsearch
    global messages
    model, docsearch, messages = chatService.client_chat_setup()
    return "success"

@app.route('/chat', methods = ['POST'])
@cross_origin()
def chat():
    global model
    global docsearch
    global messages
    
    if request.form.get('message') == None:
        return "bad request"

    if request.method == 'POST':
        if model == None or docsearch == None or messages == None:
            return {"message": "processing...\n"}
        question = request.form.get('message')
        try:
            answer = chatService.chat_llm(question=question, messages=messages, chat=model["chat"], pinecone_connection=docsearch)
            print({question: answer})
            return answer
        except Exception:
            print(messages)
            traceback.print_exc()
            model, docsearch, messages = chatService.client_chat_setup()
            return "something wrong in backend. please reload the page."
    else:
        return "should get question by post\n"