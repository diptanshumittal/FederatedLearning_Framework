import collections
from mnist_pytorch_model import create_seed_model, Net
import torch
from pytorchhelper import PytorchHelper
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


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


def validate(model, loss, settings):
    print("-- RUNNING VALIDATION --", flush=True)

    def evaluate(model, loss, dataloader):
        model.eval()
        train_loss = 0
        train_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(settings["device"]), y.to(settings["device"])
                output = model(x)
                train_loss += 32 * loss(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(y.view_as(pred)).sum().item()
            train_loss /= len(dataloader.dataset)
            train_acc = train_correct / len(dataloader.dataset)
        return float(train_loss), float(train_acc)

    trainset = read_data(trainset=True)
    testset = read_data(trainset=False)
    train_loader = DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)
    test_loader = DataLoader(testset, batch_size=settings['batch_size'], shuffle=True)
    try:
        training_loss, training_acc = evaluate(model, loss, train_loader)
        test_loss, test_acc = evaluate(model, loss, test_loader)
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


def read_data(trainset=True, data_path='data/mnist.npz'):
    pack = np.load(data_path)
    if trainset:
        X = pack['x_train']
        y = pack['y_train']
    else:
        X = pack['x_test']
        y = pack['y_test']

    X = X.astype('float32')
    y = y.astype('int64')
    X = np.expand_dims(X, 1)
    X /= 255
    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.from_numpy(y)
    dataset = TensorDataset(tensor_x, tensor_y)  # create traindatset
    return dataset


def train(model, loss, optimizer, settings):
    print("-- RUNNING TRAINING --", flush=True)
    trainset = read_data(trainset=True)
    train_loader = DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)
    model.train()
    for i in range(settings['epochs']):
        for x, y in train_loader:
            x, y = x.to(settings["device"]), y.to(settings["device"])
            optimizer.zero_grad()
            output = model(x)
            error = loss(output, y)
            error.backward()
            optimizer.step()
    print("-- TRAINING COMPLETED --", flush=True)
    return model


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    helper = PytorchHelper()
    model, loss, optimizer = create_seed_model()
    model.load_state_dict(np_to_weights(helper.load_model("weights.npz")))
    model.to(device)
    settings = {
        "batch_size": 1000,
        "epochs": 1,
        "device": device
    }
    model = train(model, loss, optimizer, settings)
    report = validate(model, loss, settings)
    model.cpu()
    helper.save_model(weights_to_np(model.state_dict()), "weights.npz")
    return report


if __name__ == "__main__":
    train_model()

