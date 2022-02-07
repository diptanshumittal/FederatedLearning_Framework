import os
import sys

sys.path.append(os.getcwd())
import yaml
import torch
import threading
import collections
from torch.utils.data import DataLoader
from helper.pytorch.pytorch_helper import PytorchHelper
from helper.pytorch.lr_scheduler import CosineAnnealingLR, MultiStepLR
from model.pytorch.pytorch_models import create_seed_model
import model.pytorch.googlenet as googlenet


def weights_to_np(weights):
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy().tolist()
    return weights_np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights


class PytorchModelTrainer:
    def __init__(self, config):
        self.helper = PytorchHelper()
        self.stop_event = threading.Event()
        self.trainer_type = config["trainer"]
        self.global_model_path = config["global_model_path"]
        self.model, self.loss, self.optimizer = create_seed_model(config["model"])
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
        # torch.cuda.set_device(int(config["cuda_device"]))
        # print(torch.cuda.current_device())
        # os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_device"]
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(config["cuda_device"])
        self.loss = self.loss.to(self.device)
        args = {
            "warm_up_epochs": int(config["model"]["warm_up_epochs"]),
            "epochs": int(config["model"]["epochs"]),
            "baseline_lr": float(config["model"]["baseline_lr"]),
            "gamma": float(config["model"]["gamma"]),
            "lr": float(config["model"]["learning_rate"]),
            "lrmilestone": list(map(int, config["model"]["lrmilestone"].split(" ")))
        }
        self.scheduler = MultiStepLR(self.optimizer, args)
        print("Device being used for training :", self.device, flush=True)
        self.train_loader = DataLoader(self.helper.read_data(config["dataset"], config["data_path"], True),
                                       batch_size=int(config['batch_size']), shuffle=True)
        self.test_loader = DataLoader(self.helper.read_data(config["dataset"], config["data_path"], False),
                                      batch_size=int(config['batch_size']), shuffle=True)

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                if self.stop_event.is_set():
                    raise ValueError("Round stop requested by the reducer!!!")
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss += self.loss(output, y).item() * x.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
            loss /= len(dataloader.dataset)
            acc = correct / len(dataloader.dataset)
        return float(loss), float(acc)

    def validate(self):
        # print("-- RUNNING VALIDATION --", flush=True)
        try:
            training_loss, training_acc = self.evaluate(self.train_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)
        except Exception as e:
            print("failed to validate the model {}".format(e), flush=True)
            raise
        report = {
            "classification_report": 'evaluated',
            "status": "pass",
            "training_loss": training_loss,
            "training_accuracy": training_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        # print("-- VALIDATION COMPLETED --", flush=True)
        return report

    def train(self, settings):
        # print("-- RUNNING TRAINING --", flush=True)
        self.model.train()
        for i in range(settings['epochs']):
            for x, y in self.train_loader:
                if self.stop_event.is_set():
                    raise ValueError("Round stop requested by the reducer!!!")
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                if isinstance(self.model, googlenet.GoogLeNet):
                    outputs, aux1, aux2 = self.model(x)
                    error = self.loss(outputs, y) + 0.3 * self.loss(aux1, y) + 0.3 * self.loss(aux2, y)
                else:
                    output = self.model(x)
                    error = self.loss(output, y)
                error.backward()
                self.optimizer.step()
            self.scheduler.step()
            # print(self.optimizer.param_groups[0]["lr"])
        # print("-- TRAINING COMPLETED --", flush=True)

    def start_round(self, round_config, stop_event):
        self.stop_event = stop_event
        try:
            self.model.load_state_dict(np_to_weights(self.helper.load_model(self.global_model_path)))
            self.model.to(self.device)
            self.train(round_config)
            report = self.validate()
            self.model.cpu()
            self.helper.save_model(weights_to_np(self.model.state_dict()), self.global_model_path)
            return report
        except Exception as e:
            print(e, flush=True)
            return {"status": "fail"}


if __name__ == "__main__":
    with open('settings/settings-common.yaml', 'r') as file:
        try:
            common_config = dict(yaml.safe_load(file))
        except yaml.YAMLError as e:
            print('Failed to read model_config from settings file', flush=True)
            raise e
    print("Setting files loaded successfully !!!")
    client_config = {"training": common_config["training"]}
    client_config["training"]["model"] = common_config["model"]
    client_config["training"]["cuda_device"] = "cuda:0"
    client_config["training"]["directory"] = "data/clients/" + "1" + "/"
    client_config["training"]["data_path"] = client_config["training"]["directory"] + "data.npz"
    client_config["training"]["global_model_path"] = client_config["training"]["directory"] + "weights.npz"
    model_trainer = PytorchModelTrainer(client_config["training"])
    model_trainer.model.to(model_trainer.device)
    stop_round_event = threading.Event()
    model_trainer.stop_event = stop_round_event
    for i in range(30):
        print(i)
        model_trainer.train({"epochs": 10})
        print("After epochs", str((i + 1) * 10), "on device", client_config["training"]["cuda_device"], "results",
              model_trainer.validate())
