import yaml
from minio import Minio
from reducer_rest_service import ReducerRestService


class Reducer:
    def __init__(self):
        """ """
        with open("settings-reducer.yaml", 'r') as file:
            try:
                fedn_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                raise e
        self.buckets = ["fedn-context"]
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
            print(self.minio_client.bucket_exists(self.buckets[0]))
            self.global_model = "initial_model.npz"
            self.minio_client.fput_object(self.buckets[0], self.global_model, self.global_model)
        except Exception as e:
            print(e)
            print("Error while setting up minio configuration")
            exit()

        config = {
            "flask_port": fedn_config["flask"]["port"],
            "global_model": self.global_model
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
