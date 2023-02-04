from flask import Flask, render_template, request
import socket



app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect("/tmp/query_server")
            sock.sendall(bytes(query, 'ascii'))
    return render_template('index.html')