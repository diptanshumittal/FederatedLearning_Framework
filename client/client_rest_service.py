import uuid
import threading
from flask import Flask, jsonify, request
import json


class ClientRestService:

    def __init__(self, config):
        print(config)
        self.model_trainer = config["model_trainer"]
        self.minio_client = config["minio_client"]
        self.port = config['flask_port']
        self.server = config["server"]
        self.global_model_path = config["global_model_path"]

    def run(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            ret = {
                'description': "This is the client"
            }
            return jsonify(ret)

        @app.route('/startround')
        def start_round():
            config = {
                "round_id": request.args.get('round_id', None),
                "bucket_name": request.args.get('bucket_name', None),
                "global_model": request.args.get('global_model', None),
                "epochs": int(request.args.get('epochs', "1"))
            }
            threading.Thread(target=self.run_round, args=(config,)).start()
            ret = {
                'status': "started"
            }
            return jsonify(ret)

        app.run(host="0.0.0.0", port=self.port)

    def run_round(self, config):
        print("Running round - ", config["round_id"], flush=True)
        self.minio_client.fget_object('fedn-context', config["global_model"], self.global_model_path)
        report = self.model_trainer.start_round({"epochs": config["epochs"]})
        print(report)
        self.minio_client.fput_object(config["bucket_name"], str(uuid.uuid4()) + ".npz", self.global_model_path)
        self.server.send_round_complete_request(config["round_id"], json.dumps(report))
        print("Round ended successfully and notification sent to server", flush=True)
