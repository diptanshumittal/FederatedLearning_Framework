import collections
from model.mnist_pytorch_model import create_seed_model, create_NASglobal_model
import torch
from helper.pytorch_helper import PytorchHelper
from torch.utils.data import DataLoader
import os
import time


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


class NasTrainer:
    def __init__(self, config):
        self.helper = PytorchHelper()
        self.global_model_path = config["global_model_path"]
        self.model, self.loss, self.optimizer = create_NASglobal_model()
        os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_device"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("NAS Training \nDevice being used for training :", self.device, flush=True)
        if config["dataset"] == "Imagenet":
            self.train_loader = DataLoader(self.helper.read_data_imagenet(data_path=config["data_path"]),
                                           batch_size=int(config['batch_size']), shuffle=True)
            self.test_loader = DataLoader(self.helper.read_data_imagenet(data_path=config["data_path"], trainset=False),
                                          batch_size=int(config['batch_size']), shuffle=True)
        else:
            self.train_loader = DataLoader(self.helper.read_data_mnist(data_path=config["data_path"]),
                                           batch_size=int(config['batch_size']), shuffle=True)
            self.test_loader = DataLoader(self.helper.read_data_mnist(data_path=config["data_path"], trainset=False),
                                          batch_size=int(config['batch_size']), shuffle=True)
        # assuming test dataset is equvalent to validation dataset for training phase

    def evaluate(self, dataloader):
        self.model.eval()
        train_loss = 0
        train_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_loss += 32 * self.loss(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(y.view_as(pred)).sum().item()
            train_loss /= len(dataloader.dataset)
            train_acc = train_correct / len(dataloader.dataset)
        return float(train_loss), float(train_acc)

    def validate(self):
        print("-- RUNNING VALIDATION --", flush=True)
        try:
            training_loss, training_acc = self.evaluate(self.train_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)
        except Exception as e:
            print("failed to validate the model {}".format(e), flush=True)
            raise
        report = {
            "classification_report": 'evaluated',
            "training_loss": training_loss,
            "training_accuracy": training_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        print("-- VALIDATION COMPLETE! --", flush=True)
        return report

    def train(self, settings):
        print("-- RUNNING TRAINING --", flush=True)
        self.model.train()
        for i in range(settings['epochs']):
            # train_metrics, step_counter = train_epoch(i, model, self.train_loader, self.test_loader, self.optimizer, self.loss)
            for x, y in self.train_loader:
                # try:
                x, y = x.to(self.device), y.to(self.device)
                # print("Loaded Data on GPU", flush=True)
                output = self.model(x)
                self.optimizer.zero_grad()
                # print("output generated", flush=True)
                error = self.loss(output, y)
                # print("Loss calculated", flush=True)
                error.backward()
                self.optimizer.step()
                # print(time.time()-pre, flush=True)
                # pre = time.time()
                # except Exception as e:
                #     print(e, flush=True)
                #     exit()
        print("-- TRAINING COMPLETED --", flush=True)

    def start_round(self, round_config):
        # self.model.load_state_dict(np_to_weights(self.helper.load_model(self.global_model_path)))
        self.model.to(self.device)
        self.train(round_config)
        report = self.validate()
        # self.model.cpu()
        # self.helper.save_model(weights_to_np(self.model.state_dict()), self.global_model_path)
        return report


class PytorchModelTrainer:
    def __init__(self, config):
        self.helper = PytorchHelper()
        self.global_model_path = config["global_model_path"]
        self.model, self.loss, self.optimizer = create_seed_model()
        os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_device"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device being used for training :", self.device, flush=True)
        if config["dataset"] == "Imagenet":
            self.train_loader = DataLoader(self.helper.read_data_imagenet(data_path=config["data_path"]),
                                           batch_size=int(config['batch_size']), shuffle=True)
            self.test_loader = DataLoader(self.helper.read_data_imagenet(data_path=config["data_path"], trainset=False),
                                          batch_size=int(config['batch_size']), shuffle=True)
        else:
            self.train_loader = DataLoader(self.helper.read_data(data_path=config["data_path"]),
                                           batch_size=int(config['batch_size']), shuffle=True)
            self.test_loader = DataLoader(self.helper.read_data(data_path=config["data_path"], trainset=False),
                                          batch_size=int(config['batch_size']), shuffle=True)
        print(len(self.train_loader))

    def evaluate(self, dataloader):
        self.model.eval()
        train_loss = 0
        train_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_loss += 32 * self.loss(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(y.view_as(pred)).sum().item()
            train_loss /= len(dataloader.dataset)
            train_acc = train_correct / len(dataloader.dataset)
        return float(train_loss), float(train_acc)

    def validate(self):
        print("-- RUNNING VALIDATION --", flush=True)
        try:
            training_loss, training_acc = self.evaluate(self.train_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)
        except Exception as e:
            print("failed to validate the model {}".format(e), flush=True)
            raise
        report = {
            "classification_report": 'evaluated',
            "training_loss": training_loss,
            "training_accuracy": training_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        print("-- VALIDATION COMPLETE! --", flush=True)
        return report

    def train(self, settings):
        print("-- RUNNING TRAINING --", flush=True)
        self.model.train()
        for i in range(settings['epochs']):
            # print("Epoch :",i)
            # pre = time.time()
            for x, y in self.train_loader:
                try:
                    x, y = x.to(self.device), y.to(self.device)
                    # print("Loaded Data on GPU", flush=True)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    # print("output generated", flush=True)
                    error = self.loss(output, y)
                    # print("Loss calculated", flush=True)
                    error.backward()
                    self.optimizer.step()
                    # print(time.time()-pre, flush=True)
                    # pre = time.time()
                except Exception as e:
                    print(e, flush=True)
                    exit()
        print("-- TRAINING COMPLETED --", flush=True)

    def start_round(self, round_config):
        self.model.load_state_dict(np_to_weights(self.helper.load_model(self.global_model_path)))
        self.model.to(self.device)
        self.train(round_config)
        report = self.validate()
        self.model.cpu()
        self.helper.save_model(weights_to_np(self.model.state_dict()), self.global_model_path)
        return report


if __name__ == "__main__":
    setup_config = {
        "data_path": "data/mnist.npz",
        "global_model_path": "weights/initial_model.npz",
        "dataset": "MNIST",
        "batch_size": 4,
        "cuda_device": "0"
    }
    modelTrainer = NasTrainer(setup_config)
    settings = {
        "epochs": 1
    }
    print(modelTrainer.start_round(settings))
