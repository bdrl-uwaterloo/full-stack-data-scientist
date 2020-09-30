from flask import Flask, render_template, request, flash, redirect
import os
from model import market_evaluator

app = Flask(__name__)
market_model = market_evaluator()

@app.route('/', methods=['GET', 'POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html')
    elif(request.method == 'POST'):
        upload_file = request.files['file']
        if uploaded_file.filename == '':
            return render_template('index.html', error='No file is selected. Please selete a file')
        if upload_file and is_proper_file(uploaded_file):
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)
            result = market_model.evaluate(file_path)
            return render_template('index.html', result = result)
            
if __name__ == "__main__":
    app.run()