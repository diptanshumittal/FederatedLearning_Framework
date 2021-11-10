#!/usr/bin/python
import threading
import time

import yaml
import subprocess


def run_container(cmd):
    subprocess.call(cmd, shell=True)


threading.Thread(target=run_container, args=("docker-compose -f docker/minio.yaml up >> miniologs.txt",), daemon=True).start()
time.sleep(5)
threading.Thread(target=run_container, args=("docker-compose -f docker/reducer.yaml up >> data/reducer/log.txt",),
                 daemon=True).start()
time.sleep(50)

try:
    for i in range(1, 6):
        with open("docker/client-gpu.yaml", 'r') as file:
            config1 = dict(yaml.safe_load(file))
        print(list(config1["services"].keys()))
        config1["services"]["client" + str(i)] = config1["services"].pop(list(config1["services"].keys())[0])
        config1["services"]["client" + str(i)]["ports"][0] = "809" + str(i) + ":809" + str(i)
        config1["services"]["client" + str(i)]["container_name"] = "client" + str(i)
        with open('docker/client-gpu.yaml', 'w') as f:
            yaml.dump(config1, f)

        with open("settings/settings-client.yaml", 'r') as file:
            config = dict(yaml.safe_load(file))
        config["client"]["port"] = 8090 + i
        config["training"]["data_path"] = "data/clients/" + str(i) + "/imagenet.npz"
        config["training"]["global_model_path"] = "data/clients/" + str(i) + "/weights.npz"
        with open('settings/settings-client.yaml', 'w') as f:
            yaml.dump(config, f)
        threading.Thread(target=run_container,
                         args=("docker-compose -f docker/client-gpu.yaml up >> data/clients/" + str(i) + "/log.txt",),
                         daemon=True).start()
        time.sleep(50)
except Exception as e:
    print(e)
