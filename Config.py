# TODO: this file should hold all the default configurations as well as the class that represents the current
#  configuration used.
#  default values for args should be listed here.

import os

from datetime import datetime

DEFAULT_DEVICE = 'cpu'
DEFAULT_BATCH_SIZE = 32


class Config:
    """Holds model hyper-params and data information.

    The config class is used to store various hyper-parameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    device = 'cpu'
    n_word_features = 2  # Number of features derived from every word in the input.
    window_size = 1
    n_features = (
                         2 * window_size + 1) * n_word_features  # Number of features used for every word in the input (including the window).
    max_length = 120  # longest sequence to parse
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 15
    lr = 0.005

    def __init__(self, args):
        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = os.path.join(self.output_path, "model.weights")
        self.eval_output = os.path.join(self.output_path, "results.txt")
        self.conll_output = os.path.join(self.output_path, "predictions.conll")
        self.log_output = os.path.join(self.output_path, "log")
        self.device = int(args.device) if args.device != 'cpu' else args.device
