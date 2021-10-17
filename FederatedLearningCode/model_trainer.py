import collections
from mnist_pytorch_model import create_seed_model
import torch
from pytorch_helper import PytorchHelper
from torch.utils.data import DataLoader


def weights_to_np(weights):
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    return weights_np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights


class PytorchModelTrainer:
    def __init__(self, config):
        self.helper = PytorchHelper()
        self.global_model_path = config["global_model_path"]
        self.model, self.loss, self.optimizer = create_seed_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device being used for training :", self.device, flush=True)
        self.train_loader = DataLoader(self.helper.read_data(data_path=config["data_path"]),
                                       batch_size=int(config['batch_size']), shuffle=True)
        self.test_loader = DataLoader(self.helper.read_data(data_path=config["data_path"], trainset=False),
                                      batch_size=int(config['batch_size']), shuffle=True)

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
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                error = self.loss(output, y)
                error.backward()
                self.optimizer.step()
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
        "batch_size": 1000
    }
    modelTrainer = PytorchModelTrainer(setup_config)
    settings = {
        "epochs": 1
    }
    modelTrainer.start_round("weights.npz", settings)
    modelTrainer.start_round("weights.npz", settings)
    modelTrainer.start_round("weights.npz", settings)
