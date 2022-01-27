#!/usr/bin/python
import threading
import time
import yaml
import subprocess
from multiprocessing import Process


def run_container(cmd):
    subprocess.call(cmd, shell=True)


def start_clients_docker():
    try:
        for i in range(1, 6):
            with open("docker/client-gpu.yaml", 'r') as file:
                config1 = dict(yaml.safe_load(file))
            print(list(config1["services"].keys()))
            config1["services"]["client" + str(i)] = config1["services"].pop(list(config1["services"].keys())[0])
            config1["services"]["client" + str(i)]["ports"][0] = "809" + str(i) + ":809" + str(i)
            config1["services"]["client" + str(i)]["container_name"] = "client" + str(i)
            config1["services"]["client" + str(i)][
                "command"] = "sh -c 'pip install -r requirements.txt && python client.py " + "data/clients/" + str(
                i) + "/settings-client.yaml'"

            with open('docker/client-gpu.yaml', 'w') as f:
                yaml.dump(config1, f)

            with open("settings/settings-client.yaml", 'r') as file:
                config = dict(yaml.safe_load(file))
            config["client"]["port"] = 8090 + i
            config["training"]["data_path"] = "data/clients/" + str(i) + "/mnist.npz"
            config["training"]["global_model_path"] = "data/clients/" + str(i) + "/weights.npz"
            with open("data/clients/" + str(i) + "/settings-client.yaml", 'w') as f:
                yaml.dump(config, f)
            threading.Thread(target=run_container,
                             args=(
                                 "docker-compose -f docker/client-gpu.yaml up >> data/clients/" + str(i) + "/log.txt",),
                             daemon=True).start()
            time.sleep(10)
    except Exception as e:
        print(e)


def start_clients():
    try:
        available_gpus = ["cuda:0", "cuda:0"]
        for i in range(1, 2):
            with open("settings/settings-client.yaml", 'r') as file:
                config = dict(yaml.safe_load(file))
            config["training"]["cuda_device"] = available_gpus[i - 1]
            config["training"]["directory"] = "data/clients/" + str(i) + "/"
            with open("settings/settings-client.yaml", 'w') as f:
                yaml.dump(config, f)
            Process(target=run_container,
                    args=("python Client/client.py >> data/clients/" + str(i) + "/log.txt",), daemon=True).start()
            time.sleep(3)
    except Exception as e:
        print(e)


start_clients()
