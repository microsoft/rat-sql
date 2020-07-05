"""Tools to save/restore model from checkpoints."""

import argparse
import shutil
import sys
import os
import re
import json
import time

import torch

CHECKPOINT_PATTERN = re.compile(r'^model_checkpoint-(\d+)$')


class ArgsDict(dict):

    def __init__(self, **kwargs):
        super(ArgsDict, self).__init__()
        for key, value in kwargs.items():
            self[key] = value
        self.__dict__ = self


def load_checkpoint(item_dict, model_dir, map_location=None, step=None):
    """ item_dict: {"model": model, "opt1": opt1, ...}"""
    path = os.path.join(model_dir, 'model_checkpoint')
    if step is not None:
        path += f'-{step:08d}'
    if os.path.exists(path):
        print("Loading model from %s" % path)
        checkpoint = torch.load(path, map_location=map_location)

        old_state_dict = item_dict["model"].state_dict()
        for key in old_state_dict.keys():
            if key not in checkpoint['model']:
                checkpoint['model'][key] = old_state_dict[key]

        for item_name in item_dict:
            item_dict[item_name].load_state_dict(checkpoint[item_name])
        return checkpoint.get('step', 0)
    return 0


def load_and_map_checkpoint(model, model_dir, remap):
    path = os.path.join(model_dir, 'model_checkpoint')
    print("Loading parameters %s from %s" % (remap.keys(), model_dir))
    checkpoint = torch.load(path)
    new_state_dict = model.state_dict()
    for name, value in remap.items():
        # TODO: smarter mapping.
        new_state_dict[name] = checkpoint['model'][value]
    model.load_state_dict(new_state_dict)


def save_checkpoint(items, step, model_dir, ignore=[],
                    keep_every_n=10000000):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path_without_step = os.path.join(model_dir, 'model_checkpoint')
    step_padded = format(step, '08d')
    state_dict = items["model"].state_dict()
    if ignore:
        for key in state_dict.keys():
            for item in ignore:
                if key.startswith(item):
                    state_dict.pop(key)
    path_with_step = f'{path_without_step}-{step_padded}'

    saved_dic = {}
    for key in items:
        saved_dic[key] = items[key].state_dict()
    torch.save({**saved_dic, "step": step}, path_with_step)

    try:
        os.unlink(path_without_step)
    except FileNotFoundError:
        pass
    try:
        os.symlink(os.path.basename(path_with_step), path_without_step)
    except OSError:
        shutil.copy2(path_with_step, path_without_step)

    # Cull old checkpoints.
    if keep_every_n is not None:
        all_checkpoints = []
        for name in os.listdir(model_dir):
            m = CHECKPOINT_PATTERN.match(name)
            if m is None or name == os.path.basename(path_with_step):
                continue
            checkpoint_step = int(m.group(1))
            all_checkpoints.append((checkpoint_step, name))
        all_checkpoints.sort()

        last_step = float('-inf')
        for checkpoint_step, name in all_checkpoints:
            if checkpoint_step - last_step >= keep_every_n:
                last_step = checkpoint_step
                continue
            os.unlink(os.path.join(model_dir, name))


class Saver(object):
    """Class to manage save and restore for the model and optimizer."""

    def __init__(self, items, keep_every_n=None):
        assert type(items) == dict
        assert "model" in items
        self._items = items
        self._keep_every_n = keep_every_n

    def restore(self, model_dir, map_location=None,
                step=None, item_keys=["model", "optimizer"]):
        """Restores model and optimizer from given directory.
            Specify what shoud be restored

        Returns:
           Last training step for the model restored.
        """
        items2restore = {k: self._items[k] for k in item_keys}
        last_step = load_checkpoint(
            items2restore, model_dir, map_location, step)
        return last_step

    def save(self, model_dir, step):
        """Saves model and optimizer to given directory.

        Args:
           model_dir: Model directory to save.
           step: Current training step.
        """
        save_checkpoint(self._items, step, model_dir,
                        keep_every_n=self._keep_every_n)

    def restore_part(self, other_model_dir, remap):
        """Restores part of the model from other directory.

        Useful to initialize part of the model with another pretrained model.

        Args:
            other_model_dir: Model directory to load from.
            remap: dict, remapping current parameters to the other model's.
        """
        load_and_map_checkpoint(self._items["model"], other_model_dir, remap)
