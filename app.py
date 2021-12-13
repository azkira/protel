#app.py
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'wav'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename('pred.wav')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif, wav')
        return redirect(request.url)

@app.route('/api', methods=['GET'])
def get_json():
    
    med = np.asarray(librosa.load(str('./static/uploads/pred.wav')))
    med = med[0]
    window_size = 1024
    window = np.hanning(window_size)

    stft = librosa.core.spectrum.stft(med, n_fft=window_size, hop_length=512, window=window)

    out = np.abs(stft) **2
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), x_coords=None, y_coords=None,  y_axis='log', x_axis='time', sr=22050, hop_length=512, fmin=None, fmax=None, bins_per_octave = 12, ax=ax)
    fig.savefig("./static/uploads/pred.jpg")

    model = tf.keras.models.load_model('model.h5')
    labels = ['food', 'brush', 'isolation']
    img = image.load_img('./static/uploads/pred.jpg', target_size=(128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediksi= model.predict(images, batch_size=0)
    confiden = '%.3f'%(np.argmax(prediksi)*100)
    catlab = labels[np.argmax(prediksi)]

    outp = {
            "indikasi": catlab,
            # "confidence": str(confiden) + '%'
        }

    return jsonify(outp)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()