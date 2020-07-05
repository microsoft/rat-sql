import attr
import math
import torch
import transformers

from ratsql.utils import registry

registry.register('optimizer', 'adadelta')(torch.optim.Adadelta)
registry.register('optimizer', 'adam')(torch.optim.Adam)
registry.register('optimizer', 'sgd')(torch.optim.SGD)


@registry.register('lr_scheduler', 'warmup_polynomial')
@attr.s
class WarmupPolynomialLRScheduler:
    param_groups = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()
    power = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                    (self.start_lr - self.end_lr) * (
                        1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                    + self.end_lr)

        for param_group in self.param_groups:
            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'warmup_polynomial_group')
@attr.s
class WarmupPolynomialLRSchedulerGroup(WarmupPolynomialLRScheduler):
    start_lrs = attr.ib()
    """
    Each param group has it's own start lr
    start lr is in the same order as param groups,
    """

    def update_lr(self, current_step):
        for start_lr, param_group in zip(self.start_lrs, self.param_groups):
            if current_step < self.num_warmup_steps:
                warmup_frac_done = current_step / self.num_warmup_steps
                new_lr = start_lr * warmup_frac_done
            else:
                new_lr = (
                        (start_lr - self.end_lr) * (
                            1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                        + self.end_lr)

            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'warmup_cosine')
@attr.s
class WarmupCosineLRScheduler:
    param_groups = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                    (self.start_lr - self.end_lr) * 0.5 * (
                        1 + math.cos(math.pi * (current_step - self.num_warmup_steps) / self.decay_steps))
                    + self.end_lr)

        for param_group in self.param_groups:
            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'noop')
class NoOpLRScheduler:
    def __init__(self, optimizer):
        pass

    def update_lr(self, current_step):
        pass


@registry.register('optimizer', 'bertAdamw')
class BertAdamW(transformers.AdamW):
    """
    Given a model and its bert module, create parameter groups with different lr
    """

    def __init__(self, non_bert_params, bert_params, lr=1e-3, bert_lr=2e-5, **kwargs):
        self.bert_param_group = {"params": bert_params, "lr": bert_lr, "weight_decay": 0}
        self.non_bert_param_group = {"params": non_bert_params}

        params = [self.non_bert_param_group, self.bert_param_group]
        if "name" in kwargs: del kwargs["name"]  # TODO: fix this
        super(BertAdamW, self).__init__(params, lr=lr, **kwargs)


@registry.register('lr_scheduler', 'bert_warmup_polynomial_group')
@attr.s
class BertWarmupPolynomialLRSchedulerGroup(WarmupPolynomialLRScheduler):
    """
    Set the lr of bert to be zero when the other param group is warming-up
    """
    start_lrs = attr.ib()

    # Bert parameters are in the second group by default
    def update_lr(self, current_step):
        for i, (start_lr, param_group) in enumerate(zip(self.start_lrs, self.param_groups)):
            if current_step < self.num_warmup_steps:
                if i == 0:
                    warmup_frac_done = current_step / self.num_warmup_steps
                    new_lr = start_lr * warmup_frac_done
                else:  # fix bert during warm-up
                    assert i == 1
                    new_lr = 0
            else:
                new_lr = (
                        (start_lr - self.end_lr) * (
                            1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                        + self.end_lr)

            param_group['lr'] = new_lr
