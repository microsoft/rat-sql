#!/usr/bin/env python

import argparse
import json

import _jsonnet
import attr
from ratsql.commands import preprocess, train, infer, eval

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval", choices=["preprocess", "train", "eval"])
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file,
                                   model_config_args, logdir)
        train.main(train_config)
    elif args.mode == "eval":
        for step in exp_config["eval_steps"]:
            infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.infer"
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            infer.main(infer_config)

            eval_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.eval"
            eval_config = EvalConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                infer_output_path,
                eval_output_path
            )
            eval.main(eval_config)

            res_json = json.load(open(eval_output_path))
            print(step, res_json['total_scores']['all']['exact'])


if __name__ == "__main__":
    main()
