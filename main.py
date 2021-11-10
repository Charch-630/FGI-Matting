import os

import toml
import argparse
from  pprint import pprint

import torch
from   torch.utils.data import DataLoader

import utils
from   utils import CONFIG
from   tester import Tester

import dataloader

def main():

    CONFIG.log.logging_path += "_test"
    
    if CONFIG.test.alpha_path is not None:
        utils.make_dir(CONFIG.test.alpha_path)
    utils.make_dir(CONFIG.log.logging_path)

    # Create a logger
    logger = utils.get_logger(CONFIG.log.logging_path,
                                logging_level=CONFIG.log.logging_level)


    test_dataloader = dataloader.get_Test_dataloader()


    tester = Tester(test_dataloader=test_dataloader)
    tester.test()
        

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--config', type=str, default='config/FGI_config.toml')

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    if CONFIG.test.alpha_path is not None:
        CONFIG.test.alpha_path = os.path.join(CONFIG.test.alpha_path, CONFIG.version)

    pprint(CONFIG)


    #Test
    main()
