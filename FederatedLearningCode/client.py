import yaml
from minio import Minio
import torch
from client_rest_service import ClientRestService
from model_trainer import PytorchModelTrainer


class Client:
    def __init__(self):
        """ """
        with open("settings-client.yaml", 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                exit()
                raise e
        print("Settings file loaded successfully !!!")
        try:
            storage_config = fedn_config["storage"]
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
            self.model_trainer = PytorchModelTrainer(fedn_config["training"])
        except Exception as e:
            print("Error in model trainer setup ", e)
            exit()
        print("Model Trainer setup successful!!!")

        try:
            config = {
                "flask_port": fedn_config["client"]["port"],
                "reducer": fedn_config["reducer"],
                "client_config": fedn_config["client"],
                "global_model_path": fedn_config["training"]["global_model_path"]
            }
            self.rest = ClientRestService(self.minio_client, self.model_trainer, config)
        except Exception as e:
            print("Error in setting up Rest Service", e)
            exit()
        print("Reducer connected successfully !!!")

    def run(self):
        print("------------Starting Rest service on Flask server----------")
        # threading.Thread(target=self.control_loop, daemon=True).start()
        self.rest.run()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit()
    try:
        client = Client()
        client.run()
    except Exception as e:
        print(e, flush=True)
