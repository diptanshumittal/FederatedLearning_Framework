import os
import sys

sys.path.append(os.getcwd())
import time
import yaml
import json
import torch
import socket
import threading
import requests as r
from minio import Minio
from contextlib import closing
from Client.client_rest_service import ClientRestService
from model.pytorch_model_trainer import PytorchModelTrainer


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    print(ip)
    s.close()
    return ip


class Server:
    def __init__(self, name, port, id):
        self.id = id
        self.name = name
        self.port = port
        self.connected = False
        self.connect_string = "http://{}:{}".format(self.name, self.port)

    def send_round_complete_request(self, round_id, report):
        try:
            retval = r.get(
                "{}?round_id={}&client_id={}&report={}".format(self.connect_string + '/roundcompletedbyclient',
                                                               round_id, self.id, report))
            if retval.json()['status'] == "Success":
                print("Round ended successfully and notification received by server successfully", flush=True)
                return True
            return False
        except Exception as e:
            return False

    def send_round_stop_request(self, round_id):
        try:
            retval = r.get(
                "{}?round_id={}&client_id={}".format(self.connect_string + '/roundstoppedbyclient',
                                                     round_id, self.id))
            if retval.json()['status'] == "Success":
                print("Round stopped successfully and notification received by server successfully", flush=True)
                return True
            return False
        except Exception as e:
            return False

    def connect_with_server(self, client_config):
        try:
            print("Trying to connect with the reducer")
            for i in range(5):
                retval = r.get("{}?name={}&port={}".format(self.connect_string + '/addclient',
                                                           client_config["hostname"], client_config["port"]))
                if retval.json()['status'] == "added":
                    self.connected = True
                    print("Connected with the reducer!!")
                    return True
                time.sleep(2)
            return False
        except Exception as e:
            self.connected = False
            return False

    def check_server(self, client_config):
        try:
            retval = r.get("{}?name={}&port={}".format(self.connect_string + '/clientcheck',
                                                       client_config["hostname"], client_config["port"]))
            if retval.json()['status'] != "Available":
                print("Server disconnected", flush=True)
                self.connected = False
                return False
            return True
        except Exception as e:
            self.connected = False
            return False


class Client:
    def __init__(self, settings_path):
        with open(settings_path, 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                exit()
        with open(os.getcwd() + '/settings/settings-common.yaml', 'r') as file:
            try:
                common_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read model_config from settings file', flush=True)
                raise e
        print("Setting files loaded successfully !!!")
        fedn_config["training"]["model"] = common_config["model"]
        fedn_config["client"] = {
            "hostname": get_local_ip(),
            "port": find_free_port()
        }
        fedn_config["training"]["data_path"] = fedn_config["training"]["directory"] + "mnist.npz"
        fedn_config["training"]["global_model_path"] = fedn_config["training"]["directory"] + "weights.npz"

        try:
            storage_config = common_config["storage"]
            assert (storage_config["storage_type"] == "S3")
            minio_config = storage_config["storage_config"]
            self.minio_client = Minio("{0}:{1}".format(minio_config["storage_hostname"], minio_config["storage_port"]),
                                      access_key=minio_config["storage_access_key"],
                                      secret_key=minio_config["storage_secret_key"],
                                      secure=minio_config["storage_secure_mode"])
            assert (self.minio_client.bucket_exists("fedn-context"))
        except Exception as e:
            print(e)
            print("Error while setting up minio configuration")
            exit()
        print("Minio client connected successfully !!!")

        try:
            response = self.minio_client.get_object('fedn-context', "reducer_config.txt")
            fedn_config["reducer"] = json.loads(response.data)["reducer"]
        except Exception as e:
            print("Error in loading reducer_config file from minio", e)
            exit()
        finally:
            response.close()
            response.release_conn()
        try:
            self.model_trainer = PytorchModelTrainer(fedn_config["training"])
        except Exception as e:
            print("Error in model trainer setup ", e)
            exit()
        print("Model Trainer setup successful!!!")

        try:
            self.id = fedn_config["client"]["hostname"] + ":" + str(fedn_config["client"]["port"])
            self.client_config = fedn_config["client"]
            self.server = Server(fedn_config["reducer"]["hostname"], fedn_config["reducer"]["port"], self.id)
            if not self.server.connect_with_server(fedn_config["client"]):
                print("here")
                raise
            config = {
                "server": self.server,
                "minio_client": self.minio_client,
                "model_trainer": self.model_trainer,
                "flask_port": fedn_config["client"]["port"],
                "global_model_path": fedn_config["training"]["global_model_path"]
            }
            self.rest = ClientRestService(config)
        except Exception as e:
            print("Error in setting up Rest Service", e)
            exit()
        print("Reducer connected successfully adn rest service started!!!")
        threading.Thread(target=self.server_health_check, daemon=True).start()

    def server_health_check(self):
        while True:
            if self.server.connected:
                if not self.server.check_server(self.client_config):
                    x = 1
                else:
                    x = 10
            else:
                if not self.server.connect_with_server(self.client_config):
                    x = 1
                else:
                    x = 10
            if not self.server.connected:
                print("Server health check status : ", self.server.connected, flush=True)
            time.sleep(x)

    def run(self):
        print("------------Starting Rest service on Flask server----------")
        # threading.Thread(target=self.control_loop, daemon=True).start()
        self.rest.run()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Cuda not available!!")
        exit()
    # print(torch.__version__, flush=True)
    try:
        if len(sys.argv) == 1:
            sys.argv.append(os.getcwd() + "/settings/settings-client.yaml")
        client = Client(sys.argv[1])
        client.run()
    except Exception as e:
        print(e, flush=True)
