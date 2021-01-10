"""
This file defines the configuration classes for the different parts of the project.
For any part that is dependent on a changing configuration (mostly: setup, train and test) the configuration should
move around using a Configuration object that is  defined in this file.

Author:
    Gal Suchetzky (galsuchetzky@gmail.com)

TODOs:
-   The Config classes for setup, train and test are already defined here.
    Edit them to support more configuration and arguments you have added in the Args.py file.
-   If needed, add more configuration classes.
"""

import os

from datetime import datetime
from Defaults import *


# class Config:
#     """Holds model hyper-params and data information.
#
#     The config class is used to store various hyper-parameters and dataset
#     information parameters. Model objects are passed a Config() object at
#     instantiation.
#     """
#     device = DEFAULT_DEVICE
#     # n_word_features = 2  # Number of features derived from every word in the input.
#     # window_size = 1
#     # n_features = (
#     #                      2 * window_size + 1) * n_word_features  # Number of features used for every word in the input (including the window).
#     # max_length = 120  # longest sequence to parse
#     # n_classes = 5
#     dropout = DEFAULT_DROPOUT
#     # embed_size = 50
#     # hidden_size = 300
#     # batch_size = 32
#     n_epochs = DEFAULT_N_EPOCHS
#     lr = DEFAULT_LR
#
#     def __init__(self, args):
#         # Define path in which to save the models.
#         if "model_path" in args:
#             self.output_path = args.model_path
#         else:
#             self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
#
#         self.model_output = os.path.join(self.output_path, "model.weights")
#         self.eval_output = os.path.join(self.output_path, "results.txt")
#         # self.conll_output = os.path.join(self.output_path, "predictions.conll")
#         self.log_output = os.path.join(self.output_path, "log")
#         self.device = int(args.device) if args.device != DEFAULT_DEVICE else args.device


class SetupConfig:
    """
    Holds configuration for the setup script.
    """

    def __init__(self, args):
        # Get the URLs for the datasets to download
        self.dataset_url = args.dataset_url
        self.data_path = args.data_path


class TrainConfig:
    """
    Holds configuration for the training process.
    """

    def __init__(self, args):
        pass


class TestConfig:
    """
    Holds configuration for the test process.
    """

    def __init__(self, args):
        pass


