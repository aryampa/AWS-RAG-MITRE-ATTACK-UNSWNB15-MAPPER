# main.py
import os
import io
import pandas as pd
from flask import Flask, request, jsonify
import awsgi2
import logging

from structured_mapper import run_cloud_csv_mapper
from structured_mapper import ping_mapper_handler
from intel_mapper import *

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "csv-api", "routes": ["/ping","/ping-mapper", "/map-csv"]})

@app.get("/ping")
def ping():
    return jsonify({"ok": True, "service": "angus-classifier", "httpapi": "v2"})

@app.get("/ping-files")
def ping_files():
    try:
        return jsonify({"ok": True, "files":os.listdir("/var/task/")})
    except Exception as e:
        logging.exception("Error in /ping-files:")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.get("/ping-mapper")
def ping_mapper():

    try:
        return ping_mapper_handler()
    except Exception as e:
        logging.exception("Error in /ping-mapper:")
        return jsonify({"status": "error", "message": str(e)}), 400



@app.post("/map-csv")
def map_csv():
    return run_cloud_csv_mapper()



#unstrucured methods

@app.get("/hello-analyser")
def _hello_analyse():
    return jsonify({"status": "ok"})


@app.get("/ping-analyser")
def _analyze_static():
    try:
        return analyze_static()
    except Exception as e:
        logging.exception("Error in /ping-analyser:")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.post("/analyse")
def _analyse():
    try:
        return analyse()
    except Exception as e:
        logging.exception("Error in /analyse:")
        return jsonify({"status": "error", "message": str(e)}), 400

#todo implement in intel_mapper.py
@app.post("/analyze-raw-text")
def _analyse_raw():
    return jsonify({"status": "ok", "message": "not yet implemented"})




def handler(event, context):
    # Works with HTTP API v2 events
    return awsgi2.response(app, event, context)