import time
import uuid
from flask import Flask, jsonify, request
import requests as r
import threading
import os
from pytorchhelper import PytorchHelper


class Client:
    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.connect_string = "http://{}:{}".format(self.name, self.port)

    def send_round_start_request(self, round_id, bucket_name, global_model):
        r.get("{}?round_id={}&bucket_name={}&global_model={}".format(self.connect_string + '/startround',
                                                                     round_id, bucket_name, global_model))


class ReducerRestService:

    def __init__(self, minio_client, config):
        print(config)
        self.minio_client = minio_client
        self.port = config['flask_port']
        self.clients = []
        self.rounds = 0
        self.global_model = config["global_model"]
        self.clients_updated = 0

    def run(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            ret = {
                'description': "This is the reducer"
            }
            return jsonify(ret)

        @app.route('/addclient')
        def add_client():
            name = request.args.get('name', None)
            port = request.args.get('port', None)
            self.clients.append(Client(name, port))
            ret = {
                'status': "added"
            }
            return jsonify(ret)

        @app.route('/training')
        def start_training():
            config = {
                "rounds": int(request.args.get('rounds', None)),
                "round_time": int(request.args.get('round_time', None))
            }
            threading.Thread(target=self.train, args=(config,)).start()
            ret = {
                'status': "Training started"
            }
            return jsonify(ret)

        @app.route('/roundcompletedbyclient')
        def round_completed_by_client():
            if self.rounds == int(request.args.get('round_id', None)):
                self.clients_updated -= 1
            ret = {
                'status': "Details updated"
            }
            return jsonify(ret)

        app.run(host="0.0.0.0", port=self.port)

    def train(self, config):
        for i in range(config["rounds"]):
            self.rounds += 1
            bucket_name = "round" + str(self.rounds)
            if self.minio_client.bucket_exists(bucket_name):
                for obj in self.minio_client.list_objects(bucket_name):
                    self.minio_client.remove_object(bucket_name, object_name=obj.object_name)
            else:
                self.minio_client.make_bucket(bucket_name)
            self.clients_updated = len(self.clients)
            for client in self.clients:
                client.send_round_start_request(self.rounds, bucket_name, self.global_model)
            round_time = 0
            while True:
                if self.clients_updated < 1 or round_time > config["round_time"]:
                    break
                round_time += 1
                time.sleep(1)

            model = None
            helper = PytorchHelper()
            processed_model = 0
            for obj in self.minio_client.list_objects(bucket_name):
                self.minio_client.fget_object(bucket_name, obj.object_name, obj.object_name)
                if processed_model == 0:
                    model = helper.load_model(obj.object_name)
                else:
                    model = helper.increment_average(model, helper.load_model(obj.object_name),
                                                     processed_model + 1)
                processed_model += 1
                os.remove(obj.object_name)

            if model:
                model_name = str(uuid.uuid4()) + ".npz"
                helper.save_model(model, model_name)
                self.minio_client.fput_object("fedn-context", model_name, model_name)
                self.global_model = model_name
                os.remove(model_name)
        print("Training for {} rounds ended with global model {}".format(str(config["rounds"]), self.global_model),
              flush=True)
