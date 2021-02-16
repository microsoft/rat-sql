import argparse
import collections
import datetime
import json
import os

import _jsonnet
import attr
import torch

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import ast_util
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql import optimizers

from ratsql.utils import registry
from ratsql.utils import random_state
from ratsql.utils import saver as saver_mod

# noinspection PyUnresolvedReferences
from ratsql.utils import vocab


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)

    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)


class Logger:
    def __init__(self, log_path=None, reopen_to_flush=False):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = f'[{datetime.datetime.now().replace(microsecond=0).isoformat()}] {msg}'
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()


class Trainer:
    def __init__(self, logger, config):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.logger = logger
        self.train_config = registry.instantiate(TrainConfig, config['train'])
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)

        self.init_random = random_state.RandomContext(self.train_config.init_seed)
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.instantiate(
                registry.lookup('model', config['model']).Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct('model', config['model'],
                                            unused_keys=('encoder_preproc', 'decoder_preproc'),
                                            preproc=self.model_preproc, device=self.device)
            self.model.to(self.device)

    def train(self, config, modeldir):
        # slight difference here vs. unrefactored train: The init_random starts over here.
        # Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            # We may be able to move optimizer and lr_scheduler to __init__ instead. Empirically it works fine. I think that's because saver.restore 
            # resets the state by calling optimizer.load_state_dict. 
            # But, if there is no saved file yet, I think this is not true, so might need to reset the optimizer manually?
            # For now, just creating it from scratch each time is safer and appears to be the same speed, but also means you have to pass in the config to train which is kind of ugly.

            # TODO: not nice
            if config["optimizer"].get("name", None) == 'bertAdamw':
                bert_params = list(self.model.encoder.bert_model.parameters())
                assert len(bert_params) > 0
                non_bert_params = []
                for name, _param in self.model.named_parameters():
                    if "bert" not in name:
                        non_bert_params.append(_param)
                assert len(non_bert_params) + len(bert_params) == len(list(self.model.parameters()))

                optimizer = registry.construct('optimizer', config['optimizer'], non_bert_params=non_bert_params,
                                               bert_params=bert_params)
                lr_scheduler = registry.construct('lr_scheduler',
                                                  config.get('lr_scheduler', {'name': 'noop'}),
                                                  param_groups=[optimizer.non_bert_param_group,
                                                                optimizer.bert_param_group])
            else:
                optimizer = registry.construct('optimizer', config['optimizer'], params=self.model.parameters())
                lr_scheduler = registry.construct('lr_scheduler',
                                                  config.get('lr_scheduler', {'name': 'noop'}),
                                                  param_groups=optimizer.param_groups)

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            {"model": self.model, "optimizer": optimizer}, keep_every_n=self.train_config.keep_every_n)
        last_step = saver.restore(modeldir, map_location=self.device)

        #lr fix to not break scheduler when loading from checkpoint
        lr_scheduler.param_groups = optimizer.param_groups

        if "pretrain" in config and last_step == 0:
            pretrain_config = config["pretrain"]
            _path = pretrain_config["pretrained_path"]
            _step = pretrain_config["checkpoint_step"]
            pretrain_step = saver.restore(_path, step=_step, map_location=self.device, item_keys=["model"])
            saver.save(modeldir, pretrain_step)  # for evaluating pretrained models
            last_step = pretrain_step

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x))
        train_eval_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=lambda x: x)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.train_config.eval_batch_size,
            collate_fn=lambda x: x)

        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self._eval_model(self.logger, self.model, last_step, train_eval_data_loader, 'train',
                                         num_eval_items=self.train_config.num_eval_items)
                    if self.train_config.eval_on_val:
                        self._eval_model(self.logger, self.model, last_step, val_data_loader, 'val',
                                         num_eval_items=self.train_config.num_eval_items)

                # Compute and apply gradient
                with self.model_random:
                    for _i in range(self.train_config.num_batch_accumulated):
                        if _i > 0:  batch = next(train_data_loader)
                        loss = self.model.compute_loss(batch)
                        norm_loss = loss / self.train_config.num_batch_accumulated
                        norm_loss.backward()

                    if self.train_config.clip_grad:
                        torch.nn.utils.clip_grad_norm_(optimizer.bert_param_group["params"], \
                                                       self.train_config.clip_grad)
                    optimizer.step()
                    lr_scheduler.update_lr(last_step)
                    optimizer.zero_grad()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    self.logger.log(f'Step {last_step}: loss={loss.item():.4f}')

                last_step += 1
                # Run saver
                if last_step == 1 or last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step)

            # Save final model
            saver.save(modeldir, last_step)

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _eval_model(logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
            for eval_batch in eval_data_loader:
                batch_res = model.eval_on_batch(eval_batch)
                for k, v in batch_res.items():
                    stats[k] += v
                if num_eval_items and stats['total'] > num_eval_items:
                    break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != 'total':
                stats[k] /= stats['total']
        if 'total' in stats:
            del stats['total']

        kv_stats = ", ".join(f"{k} = {v}" for k, v in stats.items())
        logger.log(f"Step {last_step} stats, {eval_section}: {kv_stats}")


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    # Initialize the logger
    reopen_to_flush = config.get('log', {}).get('reopen_to_flush')
    logger = Logger(os.path.join(args.logdir, 'log.txt'), reopen_to_flush)

    # Save the config info
    with open(os.path.join(args.logdir,
                           f'config-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")}.json'), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    logger.log(f'Logging to {args.logdir}')

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == '__main__':
    args = add_parser()
    main(args)
