######section 1#######
from flask import Flask, render_template, request, flash, redirect
import os
from model import chest_xray_classifier_evaluator

# create the Flask app variable 
app = Flask(__name__)
UPLOAD_FOLDER = 'static/x-ray'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
xray_model = chest_xray_classifier_evaluator()

######section 2#######
def is_image_file(fname):
    if(not '.' in fname):
        return False
    parts = fname.rsplit('.', 1)
    extension = parts[1].lower()
    return extension in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html')
    elif(request.method == 'POST'):
        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            return render_template('index.html', error='No file is selected. Please selete a file')
        
        if uploaded_file and is_image_file(uploaded_file.filename):
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)

            result = xray_model.evaluate(img_path)
            
            return render_template('index.html', path=img_path, result = result)

######section 3#######
if __name__ == "__main__":
    app.run()