import base64
import requests
from keras.models import load_model

def predict(baseurl, args):
    payload = {'args': base64.b64encode(load_model(args))}
    r = requests.post(baseurl + "/predict", data=payload)
    return r.text

