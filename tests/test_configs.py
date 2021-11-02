import unittest
import numpy as np
import torch
import argparse

from train import train, get_default_args
from src.utils import Namespace
from configs import datasets, models


class TestConfigs(unittest.TestCase):
    def test_configs(self):
        args = get_default_args()
        for dataset in datasets:
            for model in models:
                config = Namespace(dict(
                    dataset=datasets[dataset],
                    model=models[model]
                ))
                config.dataset.max_epochs = 10
                results = train(config, args, 2)
                self.assertTrue("results/train/accuracy/mean" in results)
    
if __name__ == '__main__':
    unittest.main()