import argparse
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

parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--gpu', default='None', type=str,
                    help='GPU device to be used by the client')
parser.add_argument('--client_id', default='1', type=str, help='Client_id used for accessing dataset and saving logs')
args = parser.parse_args()
print(args)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
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
    def __init__(self, args):
        with open('settings/settings-common.yaml', 'r') as file:
            try:
                common_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read model_config from settings file', flush=True)
                raise e
        self.training_id = common_config["training"]["dataset"] + "_" + common_config["model"]["model_type"] + "_" + \
                           common_config["model"]["optimizer"] + "_" + common_config["training_identifier"]["id"]
        if not os.path.exists(os.getcwd() + "/data/logs"):
            os.mkdir(os.getcwd() + "/data/logs")
        if not os.path.exists(os.getcwd() + "/data/logs/" + self.training_id):
            os.mkdir(os.getcwd() + "/data/logs/" + self.training_id)
        sys.stdout = open(os.getcwd() + "/data/logs/" + self.training_id + "/client_" + args.client_id + ".txt", "w")
        print("Setting files loaded successfully !!!")
        client_config = {"training": common_config["training"]}
        client_config["training"]["model"] = common_config["model"]
        client_config["client"] = {
            "hostname": get_local_ip(),
            "port": find_free_port()
        }
        if args.gpu != "None":
            client_config["training"]["cuda_device"] = args.gpu
        elif "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            print("CUDA_VISIBLE_DEVICES :", os.environ["CUDA_VISIBLE_DEVICES"])
            if isinstance(os.environ["CUDA_VISIBLE_DEVICES"], list):
                client_config["training"]["cuda_device"] = "cuda:" + str(os.environ["CUDA_VISIBLE_DEVICES"][0])
            elif isinstance(os.environ["CUDA_VISIBLE_DEVICES"], str):
                client_config["training"]["cuda_device"] = "cuda:" + os.environ["CUDA_VISIBLE_DEVICES"][0]
            else:
                print(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            raise ValueError("GPU device not set!!")
        client_config["training"]["directory"] = "data/clients/" + args.client_id + "/"
        client_config["training"]["data_path"] = client_config["training"]["directory"] + "data.npz"
        client_config["training"]["global_model_path"] = client_config["training"]["directory"] + "weights.npz"
        print(client_config)
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
        response = None
        try:
            response = self.minio_client.get_object('fedn-context', "reducer_config.txt")
            client_config["reducer"] = json.loads(response.data)["reducer"]
        except Exception as e:
            print("Error in loading reducer_config file from minio", e)
            exit()
        finally:
            if response is not None:
                response.close()
                response.release_conn()
        try:
            self.model_trainer = PytorchModelTrainer(client_config["training"])
        except Exception as e:
            print("Error in model trainer setup ", e)
            exit()
        print("Model Trainer setup successful!!!")

        try:
            self.id = client_config["client"]["hostname"] + ":" + str(client_config["client"]["port"])
            self.client_config = client_config["client"]
            self.server = Server(client_config["reducer"]["hostname"], client_config["reducer"]["port"], self.id)
            if not self.server.connect_with_server(client_config["client"]):
                print("here")
                raise
            config = {
                "server": self.server,
                "minio_client": self.minio_client,
                "model_trainer": self.model_trainer,
                "flask_port": client_config["client"]["port"],
                "global_model_path": client_config["training"]["global_model_path"]
            }
            self.rest = ClientRestService(config)
        except Exception as e:
            print("Error in setting up Rest Service", e)
            exit()
        print("Reducer connected successfully and rest service started!!!")
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
        client = Client(args)
        client.run()
    except Exception as e:
        print(e, flush=True)
