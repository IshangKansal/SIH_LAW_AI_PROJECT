from flask import Flask, render_template, request, jsonify
from main import *

app = Flask(__name__)

# Function to process the input

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the message from the request body
    data = request.get_json()
    message = data['message']

    # Pass the message to the processing function
    output = code(message)

    # Return the processed output as JSON
    return jsonify({'response': output})

if __name__ == '__main__':
    app.run(debug=True)
