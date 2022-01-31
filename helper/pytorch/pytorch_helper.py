# import os
# import tempfile
# from functools import reduce
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import TensorDataset


class PytorchHelper:

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + tensorDiff / n
        return w

    # def get_tmp_path(self):
    #     fd, path = tempfile.mkstemp(suffix='.npz')
    #     os.close(fd)
    #     return path

    def get_tensor_diff(self, model, base_model):
        w = OrderedDict()
        for name in model.keys():
            w[name] = model[name] - base_model[name]
        return w

    def add_base_model(self, tensordiff, base_model, learning_rate):
        w = OrderedDict()
        for name in tensordiff.keys():
            w[name] = learning_rate * tensordiff[name] + base_model[name]
        return w

    def save_model(self, weights_dict, path=None):
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **weights_dict)
        return path

    def load_model(self, path="weights.npz"):
        b = np.load(path)
        weights_np = OrderedDict()
        for i in b.files:
            weights_np[i] = b[i]
        return weights_np

    def read_data(self, dataset, data_path, trainset):
        if dataset == "Imagenet":
            return self.read_data_imagenet(data_path, trainset)
        elif dataset == "mnist":
            return self.read_data_mnist(data_path, trainset)
        elif dataset == "cifar10":
            return self.read_data_cifar10(data_path, trainset)

    def read_data_cifar10(self, data_path, trainset=True):
        pack = np.load(data_path)
        if trainset:
            X = pack['x_train']
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        X = X.astype('float32')
        y = y.astype('int64')
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    def read_data_mnist(self, data_path, trainset=True):
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
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    # def read_data_mnist(self, data_path, trainset=True):
    #     pack = np.load(data_path)
    #     if trainset:
    #         X = pack['x_train']
    #         y = pack['y_train']
    #     else:
    #         X = pack['x_test']
    #         y = pack['y_test']
    #     X = X.astype('float32')
    #     y = y.astype('int64')
    #     print(X.shape)
    #     X = np.repeat(X , 3 , axis=2).reshape(len(X) , 3,28,28)
    #     X /= 255
    #     # print(X.shape , y.shape)
    #     tensor_x = torch.Tensor(X)
    #     tensor_y = torch.from_numpy(y)
    #     dataset = TensorDataset(tensor_x, tensor_y)
    #     return dataset

    def read_data_imagenet(self, data_path, trainset=True):
        pack = np.load(data_path)
        if trainset:
            X = pack['x_train']
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        X = X.astype('float32')
        y = y.astype('int64') - 1
        X = X.reshape(X.shape[0], 3, 64, 64)
        X /= 255
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    # def load_model_from_BytesIO(self, model_bytesio):
    #     """ Load a model from a BytesIO object. """
    #     path = self.get_tmp_path()
    #     with open(path, 'wb') as fh:
    #         fh.write(model_bytesio)
    #         fh.flush()
    #     model = self.load_model(path)
    #     os.unlink(path)
    #     return model
    #
    # def serialize_model_to_BytesIO(self, model):
    #     outfile_name = self.save_model(model)
    #
    #     from io import BytesIO
    #     a = BytesIO()
    #     a.seek(0, 0)
    #     with open(outfile_name, 'rb') as f:
    #         a.write(f.read())
    #     os.unlink(outfile_name)
    #     return a
