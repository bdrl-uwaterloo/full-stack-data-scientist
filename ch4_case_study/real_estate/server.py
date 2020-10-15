######section 1#######
from flask import Flask, render_template, request, flash, redirect
import os
import sys
from inflect import engine
from model import real_estate_predictor_evaluator

# create the Flask app variable 
app = Flask(__name__)

#define directory that will store the uploaded image
UPLOAD_FOLDER = 'static/housing_info'
#allowed extensions of the uploaded files
ALLOWED_EXTENSIONS = {'csv'}

img =  os.path.join(UPLOAD_FOLDER, 'house_img.jpeg')
#create the model object
housing_model = real_estate_predictor_evaluator('random_forest')


######section 2#######
def is_csv_file(fname):
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
        if uploaded_file and is_csv_file(uploaded_file.filename):
            # save the file to the designated folder
            csv_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(csv_path)
     

            # make prediction
            result = housing_model.evaluate(csv_path)
            result_lst=[]
            num= 0
            with open('templates/result.txt', 'w') as f:
                for i in result:
                    num = num+1
                    text = ["The " +engine().ordinal(num)+ " house's predicted median price is around $ {}".format(int(i))]
                    result_lst.append(text)
                    f.writelines(text)
                    f.writelines("\n") #newline
            

            
            # render the template with the csv and result
            return render_template('index.html', path = img,result_lst =result_lst)

######section 3#######
if __name__ == "__main__":
    app.run()