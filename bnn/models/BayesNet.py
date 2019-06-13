import torch
import torch.nn as nn

class BayesNet(object):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def fit(self, x, y, samples=1):
        raise NotImplementedError()
    
    def predict(self, x, samples):
        raise NotImplementedError()
    
    def _init_optimizer(self):
        raise NotImplementedError()
    
    def cuda(self):
        self.net.cuda()
        
    def cpu(self):
        self.net.cpu()