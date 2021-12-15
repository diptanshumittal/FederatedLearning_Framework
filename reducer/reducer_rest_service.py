import time
import uuid
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
import requests as r
import threading
import os
from helper.pytorch_helper import PytorchHelper
import json
import logging
from torch.utils.tensorboard import SummaryWriter


class Client:
    def __init__(self, name, port, rounds):
        self.name = name
        self.port = port
        self.status = "Idle"
        self.connect_string = "http://{}:{}".format(self.name, self.port)
        self.last_checked = time.time()
        self.training_acc = [0] * rounds
        self.testing_acc = [0] * rounds
        self.training_loss = [0] * rounds
        self.testing_loss = [0] * rounds

    def send_round_start_request(self, round_id, bucket_name, global_model, epochs):
        try:
            retval = r.get(
                "{}?round_id={}&bucket_name={}&global_model={}&epochs={}".format(self.connect_string + '/startround',
                                                                                round_id, bucket_name, global_model,
                                                                                epochs))
            if retval.json()['status'] == "started":
                return True
            return False
        except Exception as e:
            print("Error while send_round_start_request ",e, flush=True)
            return False

    def send_round_stop_request(self):
        try:
            retval = r.get("{}".format(self.connect_string + '/stopround'))
            if retval.json()['status'] == "stopping":
                return True
            return False
        except Exception as e:
            print("Error while send_round_stop_request ",e, flush=True)
            return False

    def update_last_checked(self):
        self.last_checked = time.time()

    def get_last_checked(self):
        return time.time() - self.last_checked


class ReducerRestService:

    def __init__(self, minio_client, config):
        self.minio_client = minio_client
        self.port = config['flask_port']
        self.clients = {}
        self.rounds = 0
        self.global_model = config["global_model"]
        self.clients_updated = 0
        self.tensorboard_path = config["tensorboard_path"]
        self.training = None
        self.stop_training_event = threading.Event()
        self.status = "Idle"
        threading.Thread(target=self.remove_disconnected_clients, daemon=True).start()

    def remove_disconnected_clients(self):
        while True:
            self.clients = {client: self.clients[client] for client in self.clients if
                            self.clients[client].get_last_checked() < 50}
            print("Connected clients - ", self.clients, flush=True)
            time.sleep(60)

    def stop_training(self):
        for _, client in self.clients.items():
            if client.status == "Idle":
                continue
            for i in range(3):
                if client.send_round_stop_request():
                    break
                else:
                    time.sleep(5)
        self.stop_training_event.set()
        self.training.join()
        self.status = "Idle"
        self.rounds -= 1

    def run(self):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
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
            self.clients[name + ":" + port] = Client(name, port, self.rounds)
            ret = {
                'status': "added"
            }
            return jsonify(ret)

        @app.route('/status')
        def status_check():
            return jsonify({"status": self.status})

        @app.route('/training')
        def start_training():
            config = {
                "rounds": int(request.args.get('rounds', '1')),
                "round_time": int(request.args.get('round_time', '200')),
                "epochs": int(request.args.get('epochs', '2'))
            }
            self.stop_training_event.clear()
            self.training = threading.Thread(target=self.train, args=(config,))
            self.training.start()
            self.status = "Training"
            ret = {
                'status': "Training started"
            }
            return jsonify(ret)

        @app.route('/stoptraining')
        def stop_training_request():
            if self.training is None or self.status == "Idle":
                return jsonify({"status": "Training not running!!"})
            if self.status == "Stopping":
                return jsonify({"status": "Stop Training request already received!!"})
            threading.Thread(target=self.stop_training, daemon=True).start()
            self.status = "Stopping"
            return jsonify({"status": "Stopping"})

        @app.route('/roundcompletedbyclient')
        def round_completed_by_client():
            round_id = int(request.args.get('round_id', "-1"))
            id = request.args.get("client_id", "0")
            if self.rounds == round_id and id in self.clients:
                if not os.path.exists(self.tensorboard_path + "/" + id):
                    os.mkdir(self.tensorboard_path + "/" + id)
                writer = SummaryWriter(self.tensorboard_path + "/" + id)
                self.clients[id].status = "Idle"
                res = request.args.get("report", None)
                if res is None:
                    res = {"training_accuracy": 0, "test_accuracy": 0, "training_loss": 0, "test_loss": 0,
                           "round_time": 0}
                else:
                    res = json.loads(res)
                writer.add_scalar('training_loss', res["training_loss"], round_id)
                writer.add_scalar('test_loss', res["test_loss"], round_id)
                writer.add_scalar('training_accuracy', res["training_accuracy"], round_id)
                writer.add_scalar('test_accuracy', res["test_accuracy"], round_id)
                writer.add_scalar('round_time', res["round_time"], round_id)
                writer.close()
                self.clients[id].training_acc.append(float(res["training_accuracy"]))
                self.clients[id].testing_acc.append(float(res["test_accuracy"]))
                self.clients[id].training_loss.append(float(res["training_loss"]))
                self.clients[id].testing_loss.append(float(res["test_loss"]))
                return jsonify({'status': "Success"})
            return jsonify({'status': "Failure"})

        @app.route('/roundstoppedbyclient')
        def round_stopped_by_client():
            round_id = int(request.args.get('round_id', "-1"))
            id = request.args.get("client_id", "0")
            if self.rounds == round_id and id in self.clients:
                self.clients[id].status = "Idle"
                return jsonify({'status': "Success"})
            return jsonify({'status': "Failure"})

        @app.route('/clientcheck')
        def client_check():
            name = request.args.get('name', None)
            port = request.args.get('port', None)
            if self.clients[name + ":" + port]:
                self.clients[name + ":" + port].update_last_checked()
                ret = {
                    'status': "Available"
                }
                return jsonify(ret)
            else:
                ret = {
                    'status': "Not Available"
                }
                return jsonify(ret)

        @app.route('/creategraph')
        def create_graph():
            for key, client in self.clients.items():
                x = np.linspace(1, len(client.training_acc), len(client.training_acc))
                plt.plot(x, client.training_acc, "-b", label="Train_Acc")
                plt.plot(x, client.testing_acc, "-r", label="Test_Acc")
                plt.legend(loc="upper right")
                plt.xlabel("Rounds")
                plt.ylabel("Accuracy")
                plt.title("Rounds vs Accuracy for client : " + key)
                plt.savefig(key + '_Acc.png')
                plt.clf()
                plt.plot(x, client.training_loss, "-b", label="Train_loss")
                plt.plot(x, client.testing_loss, "-r", label="Test_loss")
                plt.legend(loc="upper right")
                plt.xlabel("Rounds")
                plt.ylabel("Loss")
                plt.title("Rounds vs Loss for client : " + key)
                plt.savefig(key + '_Loss.png')
                plt.clf()
            ret = {
                'status': "Created"
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
            for _, client in self.clients.items():
                for i in range(3):
                    if client.send_round_start_request(self.rounds, bucket_name, self.global_model, config["epochs"]):
                        client.status = "Training"
                        break
                    else:
                        time.sleep(5)
            round_time = 0
            while True:
                client_training = 0
                for _, client in self.clients.items():
                    if client.status == "Training":
                        client_training += 1
                print("Clients in Training : " + str(client_training), flush=True)
                if client_training == 0 or round_time > config["round_time"]:
                    break
                round_time += 1
                time.sleep(1)

            model = None
            helper = PytorchHelper()
            processed_model = 0
            for obj in self.minio_client.list_objects(bucket_name):
                if self.stop_training_event.is_set():
                    break
                self.minio_client.fget_object(bucket_name, obj.object_name, obj.object_name)
                if processed_model == 0:
                    model = helper.load_model(obj.object_name)
                else:
                    model = helper.increment_average(model, helper.load_model(obj.object_name),
                                                     processed_model + 1)
                processed_model += 1
                os.remove(obj.object_name)

            if model and not self.stop_training_event.is_set():
                model_name = str(uuid.uuid4()) + ".npz"
                helper.save_model(model, model_name)
                self.minio_client.fput_object("fedn-context", model_name, model_name)
                self.global_model = model_name
                os.remove(model_name)

            if self.stop_training_event.is_set():
                return
        print("Training for {} rounds ended with global model {}".format(str(config["rounds"]), self.global_model),
              flush=True)
        self.status = "Idle"
