import uuid
from clientScript import train_model
from flask import Flask, jsonify, request
import requests as r
import threading


class Server:
    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.connect_string = "http://{}:{}".format(self.name, self.port)

    def send_round_complete_request(self, round_id):
        r.get("{}?round_id={}".format(self.connect_string + '/roundcompletedbyclient', round_id))

    def connect_with_server(self, client_config):
        retval = r.get("{}?name={}&port={}".format(self.connect_string + '/addclient',
                                                   client_config["hostname"], client_config["port"]))
        if retval.json()['status'] == "added":
            return True
        return False


class ClientRestService:

    def __init__(self, minio_client, config):
        print(config)
        self.minio_client = minio_client
        self.port = config['flask_port']
        self.server = Server(config["reducer"]["hostname"], config["reducer"]["port"])
        if not self.server.connect_with_server(config["client_config"]):
            raise

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
                "global_model": request.args.get('global_model', None)
            }
            threading.Thread(target=self.run_round, args=(config,)).start()
            ret = {
                'status': "started"
            }
            return jsonify(ret)

        app.run(host="0.0.0.0", port=self.port)

    def run_round(self, config):
        print("Running round - ", config["round_id"], flush=True)
        self.minio_client.fget_object('fedn-context', config["global_model"], "weights.npz")
        print(train_model(), flush=True)
        self.minio_client.fput_object(config["bucket_name"], str(uuid.uuid4()) + ".npz", "weights.npz")
        self.server.send_round_complete_request(config["round_id"])
        print("Round ended successfully and notification sent to server", flush=True)
