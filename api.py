from flask import Flask, request, render_template
from model import setup_model, run_prediction
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from pathlib import Path
import threading

app = Flask(__name__)
app.config.from_object('config.Config')
socketio = SocketIO(app)

def process_prediction(**kwargs):
    path = kwargs.get('image', {})
    label = run_prediction(model, path)
    socketio.emit('task_completion', {'message': str(label)})

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
    model = setup_model(app.config['MODEL_PATH'])
    app.run(host="0.0.0.0", port=8000, debug=True)