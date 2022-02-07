import math


class CosineAnnealingLR(object):
    def __init__(self, optimizer, args, eta_min=0):
        print(args)
        self.optimizer = optimizer
        self.T_max = args["epochs"] + 0.1 - args["warm_up_epochs"]
        self.T_i = args["warm_up_epochs"]
        self.eta_min = eta_min
        self.gamma = args["gamma"]
        self.base_lr = args["lr"]
        self.init_lr = args["baseline_lr"]
        self.factor = self.base_lr / self.init_lr
        self.warm_up_epochs = args["warm_up_epochs"]
        self.epoch = 1
        if self.warm_up_epochs > 0:
            assert self.factor >= 1, "The target LR {:.3f} should be >= baseline_lr {:.2f}!".format(self.base_lr,
                                                                                                    self.init_lr)

    def step(self):
        if self.epoch < self.warm_up_epochs:
            lr = self.base_lr * 1 / self.factor * (
                    self.epoch * (self.factor - 1) / self.warm_up_epochs + 1)
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * (
                        1 + math.cos(math.pi * (self.epoch - self.T_i) / self.T_max)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.epoch += 1
