######section 1#######
from flask import Flask, render_template, request, flash, redirect
import os
from model import chest_xray_classifier_evaluator

# create the Flask app variable 
app = Flask(__name__)

#define directory that will store the uploaded image
UPLOAD_FOLDER = 'static/x-ray'

#allowed extensions of the uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#create the model object
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
        # user requests the root page '/' directly from browser, just render the index.html template and return to browser
        return render_template('index.html')
    elif(request.method == 'POST'):
        # user uploads a file, so get the uploaded file
        uploaded_file = request.files['file']

        # if user does not select file, return an error message
        if uploaded_file.filename == '':
            return render_template('index.html', error='No file is selected. Please selete a file')
        
        # there is a file uploaded and the file has an image file extension
        if uploaded_file and is_image_file(uploaded_file.filename):
            # save the file to the designated folder
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)

            # make prediction
            result = xray_model.evaluate(img_path)
            
            # render the template with the image and result
            return render_template('index.html', path=img_path, result = result)

######section 3#######
if __name__ == "__main__":
    app.run()