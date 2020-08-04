from flask import Flask, redirect, request, url_for
from prometheus_client import make_wsgi_app, Summary, Counter
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from pickle import load as pload
from pickle import loads as ploads

import base64
from keras.models import load_model
import sys
import os
import pandas as pd

app = Flask(__name__)

METRICS_PREFIX = os.getenv("S2I_APP_METRICS_PREFIX", "model")
KERAS_MODEL = os.getenv("KERAS_MODEL")
PREDICTION_TIME = Summary('%s_processing_seconds' % METRICS_PREFIX, 'Time spent processing predictions')
PREDICTIONS = Counter('%s_predictions_total' % METRICS_PREFIX, 'Total predictions for a given label', ['value'])
app.model = None

@app.route('/')
def index():
  return "Make a prediction by POSTing to /predict"


@app.route('/predict', methods=['POST'])
@PREDICTION_TIME.time()
def predict():
    import json
    if 'json_args' in request.form:
        args = pd.read_json(request.form['json_args'])
        if len(args.columns) == 1 and len(args.values) > 1:
            # convert to series
            args = args.squeeze()

    else:
        args = ploads(base64.b64decode(request.form['args']))
    try:
        predictions = app.model.predict(args)
        for v in predictions:
            PREDICTIONS.labels(v).inc()
        return json.dumps(predictions.tolist())
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return str(e)


try:
    import json
    from sklearn.pipeline import Pipeline
    notebook_count = 0
    pipeline_stages = []
    if(KERAS_MODEL):
        for k, v in json.load(open("stages.json", "r")):
            if notebook_count == 0:
                pipeline_stages.append((k, pload(open(v, "rb"))))
            else:
                pipeline_stages.append((k, load_model(open(v, "rb"))))
    else:
        app.model = Pipeline([(k, pload(open(v, "rb"))) for k, v in json.load(open("stages.json", "r"))])
    app.model = Pipeline(pipeline_stages)
      
except Exception as e:
    print(str(e))
    sys.exit()

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics' : make_wsgi_app()
})

if __name__ == "__main__":
    app.logger.setLevel(0)
    app.run(host='0.0.0.0', port=8080)

