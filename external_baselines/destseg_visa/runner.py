"""Reuse the tested official training loop and prediction/evaluation pipeline."""
from external_baselines.destseg_btad import runner as shared
from . import data


def train(args, categories):
    return shared.train(args, categories, backend=data)


def infer(args, categories):
    return shared.infer(args, categories, backend=data)


def evaluate_or_summarize(args, categories):
    return shared.evaluate_or_summarize(args, categories, backend=data)
