#!/usr/bin/python
import time
import threading
import subprocess


def run_container(cmd):
    subprocess.call(cmd, shell=True)


def start_reducer():
    threading.Thread(target=run_container, args=("docker-compose -f docker/minio.yaml up >> data/miniologs.txt",), daemon=True).start()
    time.sleep(5)
    threading.Thread(target=run_container, args=("docker-compose -f docker/reducer.yaml up >> data/reducer/log.txt",),daemon=True).start()
    time.sleep(5)

start_reducer()