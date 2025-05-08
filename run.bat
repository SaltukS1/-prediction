@echo off
echo Tahmin uygulaması başlatılıyor...
cd src
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from app import app
app.run(debug=True, use_reloader=False)
"
pause 