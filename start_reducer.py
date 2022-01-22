#!/usr/bin/python
import os
import time
import socket
import threading
import subprocess

import yaml


def run_container(cmd):
    subprocess.call(cmd, shell=True)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def start_reducer_old():
    threading.Thread(target=run_container, args=("docker-compose -f docker/minio.yaml up",), daemon=True).start()
    time.sleep(5)
    threading.Thread(target=run_container, args=("docker-compose -f docker/reducer.yaml up >> data/reducer/log.txt",),
                     daemon=True).start()
    time.sleep(5)


def start_reducer():
    if not os.path.exists('data/reducer'):
        os.mkdir('data/reducer')
    if not os.path.exists('data/minio_logs'):
        os.mkdir('data/minio_logs')
    threading.Thread(target=run_container,
                     args=("./minio server minio_data/ --console-address \":9001\" >> data/minio_logs/minio_logs.txt",),
                     daemon=True).start()
    time.sleep(5)
    with open("settings/settings-common.yaml", 'r') as file:
        config = dict(yaml.safe_load(file))
    config["storage"]["storage_config"]["storage_hostname"] = get_local_ip()
    with open("settings/settings-common.yaml", 'w') as f:
        yaml.dump(config, f)
    threading.Thread(target=run_container, args=("python Reducer/reducer.py >> data/reducer/log.txt",),
                     daemon=True).start()
    time.sleep(5)


start_reducer()
