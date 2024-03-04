# Setup Hezner cloud pc

```
# Hezner cloud pc  1 VCPU 2 GB RAM 20 GB DISK LOCAL 0.01 USAGE 0/20 TB TRAFFIC OUT 4.01 price/month
# Ubuntu 22.04 LTS

# Configuration after boot
apt update
apt install python3.10-venv
python3 -m venv env
source env/bin/activate
apt install python3
apt-get install ffmpeg libsm6 libxext6
pip install fastapi uvicorn python-multipart
pip install ultralytics
pip install gunicorn
```

# Deploy
```
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

## Utils

```
# get an image
wget https://ultralytics.com/images/bus.jpg

# Esempi di curl
# use server change localhost with server_public ip
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@bus.jpg"
```                                                                                                                             
