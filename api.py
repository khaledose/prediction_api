from flask import Flask, request, render_template
from flask_socketio import SocketIO
import numpy as np
import joblib
import threading
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from PIL import Image

app = Flask(__name__)
app.config.from_object('config.Config')
socketio = SocketIO(app)

def preprocess_image(file):
    img = Image.open(file).convert('L')
    img = img.resize((4000,4000))
    img = np.array(img)
    img = np.reshape(img, (1,-1))
    img = img/255.0
    # img = cv2.imread(file, 0)
    # img = cv2.resize(img, (4000,4000))
    # img = img.reshape(1,-1)/255
    return img

def process_prediction(**kwargs):
    path = kwargs.get('image', {})
    img = preprocess_image(path)

    prediction = model.predict(img)

    label = str(np.squeeze(prediction))
    if label=='10': 
        label='0'
    
    socketio.emit('task_completion', {'message': f'processing result: {label}'})

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    if not file: 
        return render_template('index.html', label="No file")
    
    filename = secure_filename(file.filename)
    uploads_dir = Path(app.config['UPLOADS_DIR'])
    if not uploads_dir.exists():
        uploads_dir.mkdir()
    
    img_path = uploads_dir.joinpath(filename)
    file.save(img_path)

    thread = threading.Thread(target=process_prediction, kwargs={
                    'image': img_path.as_posix()})
    thread.start()
    return render_template('index.html', label='Prediction process is running')

if __name__ == "__main__":
    model = joblib.load(app.config['MODEL_PATH'])
    app.run(host="0.0.0.0", port=8000, debug=True)