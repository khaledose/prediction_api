import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    UPLOADS_DIR = os.getenv('UPLOADS_DIR', './uploads')
    MODEL_PATH = os.getenv('MODEL_PATH', 'svm_pso_model.pkl')
