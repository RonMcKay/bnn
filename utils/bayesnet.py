# Standard Library
import logging
import math
import operator
from typing import Callable, Optional, Sequence, Union
import warnings

# Firstparty libraries
from bnn.utils.general import KLLoss

# Thirdparty libraries
import torch
from torch.cuda._utils import _get_device_index
import torch.cuda.comm as comm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply, replicate, scatter
from torch.nn.parameter import Parameter
import torch.optim as optim


def entropy(input, dim=1, eps=1e-16):
    input = input + eps
    entropy = (
        (input * input.log())
        .sum(dim)
        .div(torch.log(torch.tensor(input.shape[dim], dtype=torch.float)))
        .neg()
    )
    return entropy


class BayesNetWrapper(object):
    def __init__(
        self,
        net: nn.Module,
        cuda: bool = True,
        parallel: bool = False,
        device_ids: Optional[Sequence[int]] = None,
        output_device: Optional[int] = None,
        learning_rate: float = 1e-4,
        scheduling: bool = False,
        crit: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, float, int]],
            torch.Tensor,
        ] = KLLoss(),
        weight_decay: float = 0.0,
        task: str = "classification",
    ):
        super().__init__()
        self.log = logging.getLogger(__name__ + ".BayesNetWrapper")
        self.net = net
        self.is_cuda = cuda
        self.is_parallel = parallel
        self.lr = learning_rate
        possible_tasks = ["classification", "regression"]
        if task not in possible_tasks:
            self.log.error(
                "Expected task to be one of ["
                + ",".join(possible_tasks)
                + "], but received "
                + task
            )
            raise ValueError(
                "Expected task to be one of ["
                + ",".join(possible_tasks)
                + "], but received "
                + task
            )
        self.task = task
        if cuda:
            self.cuda(device=device_ids[0] if device_ids is not None else None)
        if parallel:
            self.parallel(device_ids, output_device)

        self.crit = crit
        self._init_optimizer(weight_decay=weight_decay, scheduling=scheduling)

    def fit(self, x, y, batch_weight, samples=1, **kwargs):
        self.net.train()
        self.optimizer.zero_grad()
        if self.is_cuda:
            x, y = x.cuda(), y.cuda()
        if not self.is_parallel:
            outputs = []
            kl_total = torch.tensor(0.0, device=x.device, requires_grad=True)
            for _ in range(samples):
                out, kl = self.net(x)
                outputs.append(out)
                kl_total = kl_total + kl
            outputs = torch.stack(outputs)
            kl_total = kl_total / samples
        else:
            outputs, kl_total = self.net(x, samples=samples)

        loss = self.crit(outputs, y, kl_total, batch_weight, **kwargs)
        loss.backward()
        self.optimizer.step()

        if self.task == "classification":
            pred = F.softmax(outputs.data.cpu(), 2).mean(0).argmax(1)
            acc = (pred == y.cpu()).float().mean().item()
        elif self.task == "regression":
            acc = 0

        if "_run" in kwargs:
            # automatic logging for sacred users if _run instance is supplied
            kwargs["_run"].log_scalar("batch.loss", loss.item())
            kwargs["_run"].log_scalar("batch.accuracy", acc)

        return loss.item(), acc

    def predict(self, x, samples=10, return_on_cpu=True):
        with torch.no_grad():
            self.net.eval()
            if self.is_cuda:
                x = x.cuda()

            if not self.is_parallel:
                outputs = []
                additional_outputs = []
                for _ in range(samples):
                    out, *additional = self.net(x)
                    if return_on_cpu:
                        outputs.append(out.data.cpu())
                        additional_outputs.append([i.data.cpu() for i in additional])
                    else:
                        outputs.append(out)
                        additional_outputs.append(tuple(additional))
                outputs = torch.stack(outputs)
                add_outs = []
                for i in range(len(additional_outputs[0])):
                    add_outs.append(
                        torch.stack([j[i] for j in additional_outputs]).mean(0)
                    )
            else:
                outputs, _ = self.net(x, samples=samples)
                if return_on_cpu:
                    outputs = outputs.data.cpu()

            if self.task == "classification":
                outputs = F.softmax(outputs, 2)
                aleatoric_uncertainty = entropy(outputs, dim=2).mean(0)
                outputs = outputs.mean(0)
                uncertainty = entropy(outputs)
                epistemic_uncertainty = uncertainty - aleatoric_uncertainty
                return (outputs, aleatoric_uncertainty, epistemic_uncertainty, *add_outs)
            elif self.task == "regression":
                pred = outputs.mean(0)
                uncertainty = outputs.std(0)
                return (pred, uncertainty, *add_outs)

    def _init_optimizer(self, weight_decay, scheduling):
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=weight_decay
        )

        if scheduling:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=1.0
            )

    def cuda(self, device=None):
        if not torch.cuda.is_available():
            self.log.error("There is no cuda device available.")
            raise ConnectionAbortedError("There is no cuda device available.")
        self.log.debug("Moving model to gpu")
        self.net = self.net.cuda(device=device)
        self.is_cuda = True

    def cpu(self):
        self.log.debug("Moving model to cpu")
        self.net = self.net.cpu()
        self.is_cuda = False

    def sequential(self):
        if isinstance(self.net, ParallelSamplingWrapper):
            self.log.debug("Model is getting serialized")
            self.net = self.net.module
            self.is_parallel = False

    def parallel(self, device_ids, output_device):
        if not isinstance(self.net, ParallelSamplingWrapper):
            self.log.debug("Model is getting parallelized")
            self.net = ParallelSamplingWrapper(
                self.net, device_ids=device_ids, output_device=output_device
            )
            self.is_parallel = True

    def save(self, filename):
        if not self.is_parallel:
            torch.save(self.net.state_dict(), filename)
        else:
            torch.save(self.net.module.state_dict(), filename)
        self.log.debug("Model saved to '{}'".format(filename))

    def load(self, filename):
        if not self.is_parallel:
            self.net.load_state_dict(
                torch.load(
                    filename, map_location=None if self.is_cuda else torch.device("cpu")
                )
            )
        else:
            self.net.module.load_state_dict(
                torch.load(
                    filename, map_location=None if self.is_cuda else torch.device("cpu")
                )
            )


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

    def forward(self, x, **kwargs):
        if not self.device_ids:
            return self.module(x, **kwargs)
        inputs, kwargs = self.broadcast(x, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        replicas = self.replicate(self.module, self.device_ids)

        return self.sample(replicas, inputs, kwargs)

    def sample(self, replicas, inputs, kwargs):
        if "samples" in kwargs:
            samples = kwargs["samples"]
        else:
            samples = len(self.device_ids)
        outputs = []
        kls = []
        for _ in range(math.ceil(samples / len(self.device_ids))):
            out = self.parallel_apply(replicas, inputs, kwargs)
            out, kl = zip(*out)
            out = [x.unsqueeze(0) for x in out]
            kl = [x.unsqueeze(0) for x in kl]
            outputs = outputs + out
            kls = kls + kl
        outputs = self.gather(outputs, self.output_device)[:samples]
        kls = self.gather(kls, self.output_device)[:samples].sum().div(samples)
        return outputs, kls

    @staticmethod
    def broadcast(x, kwargs, device_ids):
        inputs = comm.broadcast(x, device_ids) if x is not None else []
        kwargs = scatter(kwargs, device_ids) if kwargs else []
        return inputs, kwargs

    @staticmethod
    def replicate(module, device_ids):
        return replicate(module, device_ids)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )

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
            warnings.warn(
                imbalance_warn.format(device_ids[min_pos], device_ids[max_pos])
            )
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return
