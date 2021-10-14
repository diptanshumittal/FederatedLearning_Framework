import yaml
from minio import Minio
import torch
from clientrestservice import ClientRestService


class Client:
    def __init__(self):
        """ """
        with open("settings-client.yaml", 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
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
            print("here")
            config = {
                "flask_port": fedn_config["client"]["port"],
                "reducer": fedn_config["reducer"],
                "client_config": fedn_config["client"]
            }
            self.rest = ClientRestService(self.minio_client, config)
        except Exception as e:
            print(e)
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
