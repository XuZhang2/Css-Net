import math
import pytorch_warmup as warmup
import torch
from torch.optim import Adam, SGD#, RAdam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizers(models, config):
    optimizer = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    momentum = config['momentum']

    parameterized_models = {key: model for key, model in models.items() if len(list(model.parameters())) > 0}

    optimizers = {}
    if optimizer == 'Adam':
        optimiers = {}
        for key, model in parameterized_models.items():
            if key == 'text_encoder':
                tmp_lr = lr/100
            elif key == 'text_transformation':
                tmp_lr = lr/10
            elif key == 'image_classifier':
                tmp_lr = lr
            elif 'encoder' in key:
                tmp_lr = lr/10
            else:
                tmp_lr = lr/10
            optimizers[key] = Adam(model.parameters(), lr=tmp_lr, weight_decay=weight_decay, amsgrad=False)
    elif optimizer == 'SGD':
        optimizers = {key: SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
                      for key, model in parameterized_models.items()}
    elif optimizer == 'RAdam':
        optimiers = {}
        for key, model in parameterized_models.items():
            if key == 'text_encoder':
                tmp_lr = lr/100
            elif key == 'text_transformation':
                tmp_lr = lr/10
            elif key == 'image_classifier':
                tmp_lr = lr
            elif 'encoder' in key:
                tmp_lr = lr/10
            else:
                tmp_lr = lr/10
            #optimizers[key] = RAdam(model.parameters(), lr=tmp_lr, weight_decay=weight_decay)
            optimizers[key] = RAdam_rewrite(model.parameters(), lr=tmp_lr, weight_decay=weight_decay)
    return optimizers


def create_lr_schedulers(optimizers, config):
    decay_step = config['decay_step']
    epoch = config['epoch']-config['warmup_period']
    gamma = config['gamma']
    if config['scheduler'] == 'steplr':
        return {key: StepLR(optimizer, step_size=decay_step, gamma=gamma) for key, optimizer in optimizers.items()}
    elif config['scheduler'] == 'cosinelr':
        return {key: CosineAnnealingLR(optimizer, epoch) for key, optimizer in optimizers.items()}

def create_warmup_schedulrers(optimizers, config):
    #warmup_period = config['warmup_period']
    #warmup_schedulers = {key: warmup.LinearWarmup(optimizer, warmup_period) for key, optimizer in optimizers.items()}
    #return warmup_schedulers
    return None

class RAdam_rewrite(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam_rewrite, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
