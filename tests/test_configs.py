import unittest
import numpy as np
import torch
import argparse

from train_classifier import train, get_default_args
from src.utils import Namespace
from configs import configs


class TestConfigs(unittest.TestCase):
    def test_configs(self):
        args = get_default_args()
        for config_name in configs:
            config = Namespace(configs[config_name])
            config.dataset.max_epochs = 10
            results = train(config, args, 2)
            self.assertTrue("results/train/accuracy/mean" in results)
    
if __name__ == '__main__':
    unittest.main()