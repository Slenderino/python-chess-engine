python-3.12.6-embed-amd64\python.exe get-pip.py
set PATH=%PATH%;%~dp0\python-3.12.6-embed-amd64\Scripts\
python-3.12.6-embed-amd64\Scripts\pip.exe install -r requirements.txt
python-3.12.6-embed-amd64\python.exe main.py
