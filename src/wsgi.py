# TensorFlow hatalarını bastır
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from waitress import serve
import importlib.util
import sys

# app.py dosyasını manuel olarak import edelim
spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), "app.py"))
app_module = importlib.util.module_from_spec(spec)
sys.modules["app"] = app_module
spec.loader.exec_module(app_module)

app = app_module.app

if __name__ == '__main__':
    print('Tahmin uygulaması Waitress WSGI sunucusu ile başlatılıyor...')
    print('Ctrl+C ile kapatabilirsiniz.')
    serve(app, host='127.0.0.1', port=5000) 