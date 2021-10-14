import time
import uuid
from clientScript import train_model
from flask import Flask, jsonify, request
import requests as r
import threading

ALLOWED_EXTENSIONS = {'gz', 'bz2', 'tar', 'zip', 'tgz'}


class Client:
    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.connect_string = "http://{}:{}".format(self.name, self.port)

    def send_round_start_request(self, round_id, bucket_name, global_model):
        r.get("{}?round_id={}&bucket_name={}&global_model={}".format(self.connect_string + '/startround',
                                                                     round_id, bucket_name, global_model))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class ReducerRestService:

    def __init__(self, minio_client, config):
        print(config)
        self.minio_client = minio_client
        self.port = config['flask_port']
        self.clients = []
        self.rounds = 0
        self.global_model = config["global_model"]

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

        app.run(host="0.0.0.0", port=self.port)

    def train(self, config):
        for i in range(config["rounds"]):
            self.rounds += 1
            bucket_name = "round" + str(self.rounds)
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
            for client in self.clients:
                client.send_round_start_request(self.rounds, bucket_name, self.global_model)
            time.sleep(config["round_time"])
            objects = self.minio_client.list_objects(bucket_name)
            print(objects)


class ClientRestService:

    def __init__(self, minio_client, config):
        print(config)
        self.minio_client = minio_client
        self.port = config['flask_port']

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
                "global_model": request.args.get('global_model', None),
                "bucket_name": request.args.get('bucket_name', None)
            }
            threading.Thread(target=self.run_round, args=(config,)).start()
            ret = {
                'status': "started"
            }
            return jsonify(ret)

        app.run(host="0.0.0.0", port=self.port)

    def run_round(self, config):
        self.minio_client.fget_object("fedn-context", config["global_model"], "weights.npz")
        train_model()
        self.minio_client.fput_object(config["bucket_name"], str(uuid.uuid4()) + ".npz", "weights.npz")
