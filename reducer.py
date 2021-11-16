import yaml
import os
import threading
from minio import Minio
import subprocess
from helper.pytorch_helper import PytorchHelper
from model.pytorch_models import create_seed_model
from pytorch_model_trainer import weights_to_np
from reducer.reducer_rest_service import ReducerRestService


def run_tensorboard(config):
    subprocess.call("fuser -k " + str(config["port"]) + "/tcp", shell=True)
    cmd = "tensorboard --host 0.0.0.0 --logdir=" + config["path"] + " --port " + str(config["port"])
    subprocess.call(cmd, shell=True)


class Reducer:
    def __init__(self):
        """ """
        with open("settings/settings-reducer.yaml", 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                raise e
        with open("settings/settings-model.yaml", 'r') as file:
            try:
                model_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read model_config from settings file', flush=True)
                raise e
        self.buckets = ["fedn-context"]
        if not os.path.exists(fedn_config["tensorboard"]["path"]):
            os.mkdir(fedn_config["tensorboard"]["path"])
        threading.Thread(target=run_tensorboard, args=(fedn_config["tensorboard"],),
                         daemon=True).start()
        try:
            if not os.path.exists('data/reducer'):
                os.mkdir('data/reducer')
            self.global_model = "initial_model.npz"
            self.global_model_path = "data/reducer/initial_model.npz"
            model, loss, optimizer = create_seed_model(model_config["model"])
            helper = PytorchHelper()
            helper.save_model(weights_to_np(model.state_dict()), self.global_model_path)
        except Exception as e:
            print("Error while creating seed model : ", e)
            raise e
        print("Seed model created successfully !!")
        try:
            storage_config = fedn_config["storage"]
            assert (storage_config["storage_type"] == "S3")
            minio_config = storage_config["storage_config"]
            self.minio_client = Minio("{0}:{1}".format(minio_config["storage_hostname"], minio_config["storage_port"]),
                                      access_key=minio_config["storage_access_key"],
                                      secret_key=minio_config["storage_secret_key"],
                                      secure=minio_config["storage_secure_mode"])
            for bucket in self.buckets:
                if not self.minio_client.bucket_exists(bucket):
                    self.minio_client.make_bucket(bucket)
            self.minio_client.fput_object(self.buckets[0], self.global_model, self.global_model_path)
        except Exception as e:
            print(e)
            print("Error while setting up minio configuration")
            exit()

        config = {
            "flask_port": fedn_config["flask"]["port"],
            "global_model": self.global_model,
            "tensorboard_path": fedn_config["tensorboard"]["path"]
        }
        self.rest = ReducerRestService(self.minio_client, config)

    def run(self):
        # threading.Thread(target=self.control_loop, daemon=True).start()
        self.rest.run()


if __name__ == "__main__":
    try:
        reducer = Reducer()
        reducer.run()
    except Exception as e:
        print(e, flush=True)
