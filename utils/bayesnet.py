import torch
import math
import operator
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import replicate, parallel_apply, scatter
from torch.cuda._utils import _get_device_index
import torch.cuda.comm as comm
import torch.nn.functional as F
from bnn.utils import KLLoss
import logging

def get_entropy(input, dim=1):
    return (input * input.log()).sum(dim)

class BayesNetWrapper(object):
    def __init__(self, net, cuda=True, parallel=False, device_ids=None, output_device=None,
                 learning_rate=0.001, ignore_index=-100, loss_weights=None, weight_decay=0, loss_reduction='sum',
                 task='classification'):
        super().__init__()
        self.log = logging.getLogger(__name__ + '.BayesNetWrapper')
        self.net = net
        self.is_cuda = cuda
        self.is_parallel = parallel
        self.lr = learning_rate
        possible_tasks = ['classification', 'regression']
        if task not in possible_tasks:
            self.log.error('Expected task to be one of [' + ','.join(possible_tasks) + '], but received ' + task)
            raise ValueError('Expected task to be one of [' + ','.join(possible_tasks) + '], but received ' + task)
        self.task = task
        if cuda:
            self.cuda()
        if parallel:
            self.parallel(device_ids, output_device)
            
        self.crit = KLLoss(ignore_index=ignore_index, weight=loss_weights, reduction=loss_reduction)
        self._init_optimizer(weight_decay=weight_decay)   
                
    def fit(self, x, y, batch_weight, samples=1):
        self.net.train()
        self.optimizer.zero_grad()
        if self.is_cuda:
            x, y = x.cuda(), y.cuda()
        if not self.is_parallel:
            outputs = []
            kl_total = 0
            for _ in range(samples):
                out, kl = self.net(x)
                outputs.append(out)
                kl_total += kl
            outputs = torch.stack(outputs)
            kl_total /= samples
        else:
            outputs, kl_total = self.net(x, samples=samples)
        
        loss = self.crit(outputs, y, kl_total, batch_weight)
        loss.backward()
        self.optimizer.step()
        
        if self.task == 'classification':
            pred = F.softmax(outputs.data.cpu(), 2).mean(0).argmax(1)
            acc = (pred == y.cpu()).float().mean().item()
        elif self.task == 'regression':
            acc = 0
        
        return loss.item(), acc
            
    def predict(self, x, samples=10, return_on_cpu=True):
        self.net.eval()
        if self.is_cuda:
            x = x.cuda()
        
        if not self.is_parallel:
            outputs = []
            for _ in range(samples):
                out, _ = self.net(x)
                if return_on_cpu:
                    outputs.append(out.data.cpu())
                else:
                    outputs.append(out)
            outputs = torch.stack(outputs)
        else:
            outputs, _ = self.net(x, samples=samples)
            if return_on_cpu:
                outputs = outputs.data.cpu()
                
        if self.task == 'classification':
            outputs = F.softmax(outputs, 2).mean(0)
            pred = outputs.argmax(1)
            uncertainty = get_entropy(outputs)
        elif self.task == 'regression':
            pred = outputs.mean(0)
            uncertainty = outputs.std(0)
        
        return pred, uncertainty
            
    def _init_optimizer(self, weight_decay):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    
    def cuda(self):
        if not torch.cuda.is_available():
            self.log.error('There is no cuda device available.')
            raise ConnectionAbortedError('There is no cuda device available.')
        self.log.debug('Moving model to gpu')
        self.net = self.net.cuda()
        self.is_cuda = True
        
    def cpu(self):
        self.log.debug('Moving model to cpu')
        self.net = self.net.cpu()
        self.is_cuda = False
        
    def sequential(self):
        if isinstance(self.net, ParallelSamplingWrapper):
            self.log.debug('Model is getting serialized')
            self.net = self.net.module
            self.is_parallel = False
            
    def parallel(self, device_ids, output_device):
        if not isinstance(self.net, ParallelSamplingWrapper):
            self.log.debug('Model is getting parallelized')
            self.net = ParallelSamplingWrapper(self.net, device_ids=device_ids, output_device=output_device)
            self.is_parallel = True
        
    def save(self, filename):
        if not self.is_parallel:
            torch.save(self.net.state_dict(), filename)
        else:
            torch.save(self.net.module.state_dict(), filename)
        self.log.debug('Model saved to \'{}\''.format(filename))
        
    def load(self, filename):
        if not self.is_parallel:
            self.net.load_state_dict(torch.load(filename))
        else:
            self.net.module.load_state_dict(torch.load(filename))
        self.log.debug('Model loaded from \'{}\''.format(filename))
        
class ParallelSamplingWrapper(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
            
        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        
        _check_balance(self.device_ids)
        
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])
            
    def forward(self, input, **kwargs):
        if not self.device_ids:
            return self.module(input, **kwargs)
        inputs, kwargs = self.broadcast(input, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        replicas = self.replicate(self.module, self.device_ids)
        
        return self.sample(replicas, inputs, kwargs)
            
    def sample(self, replicas, inputs, kwargs):
        if 'samples' in kwargs:
            samples = kwargs['samples']
        else:
            samples = len(self.device_ids)
        outputs = []
        kls = []
        for _ in range(math.ceil(samples/len(self.device_ids))):
            out = self.parallel_apply(replicas, inputs, kwargs)
            out, kl = zip(*out)
            out = [x.unsqueeze(0) for x in out]
            kl = [x.unsqueeze(0) for x in kl]
#             outputs.append(self.gather(out, self.output_device))
#             kls.append(self.gather(kl, self.output_device))
            outputs += out
            kls += kl
        outputs = self.gather(outputs, self.output_device)[:samples]
        kls = self.gather(kls, self.output_device)[:samples].sum().div(samples)
#         outputs = torch.cat(outputs)[:samples]
#         kls = torch.cat(kls)[:samples].sum().div(samples)
        return outputs, kls
        
    def broadcast(self, input, kwargs, device_ids):
        inputs = comm.broadcast(input, device_ids) if input is not None else []
        kwargs = scatter(kwargs, device_ids) if kwargs else []
        return inputs, kwargs
    
    def replicate(self, module, device_ids):
        return replicate(module, device_ids)
    
    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
    
    def gather(self, outputs, output_device):
        return comm.gather(outputs, dim=self.dim, destination=output_device)
    
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return