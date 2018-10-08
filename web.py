# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# web = Flask(__name__)
#
#
#
#
#
# @web.route('/hello')
# def home():
#     return render_template('home.html')
#
#
# @web.route('/hello', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(secure_filename(f.filename))
#         return 'file uploaded successfully'
#
#
# if __name__ == '__main__':
#   web.run (debug=True)
#   web.run(host='0.0.0.0', port=5000)

###############################################################################################################################################
import os
import keras
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
model = keras.applications.mobilenet.MobileNet(weights = None)
web = Flask(__name__)


UPLOAD_FOLDER = os.path.basename('upload')
web.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(path):
    img = Image.open(path).resize((224,224))
    img = np.array(img).reshape((1,224,224,3))
    return model.predict(img)



@web.route('/hello')
def hello_world():
    return render_template('home.html')
@web.route('/upload', methods=['POST'])
def upload_file():

    file = request.files['image']
    f = os.path.join(web.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)

    return render_template('home.html')

@web.route('/', methods=['GET','POST'])
def uploads_file():
    if request.method == 'POST':
        file = request.files['file']
        print("hello")
        path = 'img.jpg'
        file.save(path)
        prd = predict(path)
        return render_template('home.html', answer=prd)
    else:
        return render_template('home.html')

if __name__ == '__main__':
  web.run(debug=True)
  web.run(host='0.0.0.0', port=5000)
