"""
Download and pre-process the dataset.

Usage:
    > conda activate <your env name>
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
Edited by:
    Gal Suchetzky (galsuchetzky@gmail.com)
"""

import os
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from tqdm import tqdm
from zipfile import ZipFile

# TODO: Add to this file any preprocess logic for your dataset or anything else.
# Note: This file will only run once before starting to work on the project, so make sure to include here anything
# that is only required to be ran once, for example all code for preprocessing the dataset.


def download_url(url, output_path, show_progress=True):
    """
    Downloads a file from the specified URL.
    Args:
        url: The URL of the file to download.
        output_path: The path in which to save the downloaded file.
        show_progress: Weather to show a downloading progress bar.
    """

    class DownloadProgressBar(tqdm):
        """
        Customized progress bar based on the tqdm progress bar.
        """

        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    """
    Convert the given data URL to a file path in the data folder.
    Args:
        url: The URL to convert.

    Returns: The path of the data file.
    """
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    """
    Downloads the required data according to the urls given in the arguments.
    Args:
        args: the command line arguments.
    """
    downloads = [

        # TODO: add your downloads here in the following format:
        # ('name', args.<arg>_url),

    ]

    # Download the files in the downloads list.
    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    # TODO: add here any other downloading logic, for example: spacy language model


def save(filename, obj, message=None):
    """
    Saves a file to a json file.
    Args:
        filename: The name of the output saved file.
        obj: The object to save.
        message: The message to print to the console.

    """
    if message is not None:
        print(f"Saving {message}...")

    with open(filename, "w") as fh:
        json.dump(obj, fh)


def pre_process(args):
    """
    Preprocess your dataset.
    Args:
        args: all arguments recived by the user as well ad the default ones.
    """
    # TODO: add your preprocess code here.
    pass

    # Note: after preprocessing, you can save the preprocessed results in a json file using the save function.
    # example:
    # save(args.word_emb_file, word_emb_mat, message="word embedding")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Preprocess dataset
    # TODO: add any required paths for preprocess to args_.

    if args_.include_test_examples:  # If should include the test examples in the preprocess.
        args_.test_file = url_to_data_path(args_.test_url)

    # TODO: add to args_ anything else required for the preprocessing of your data.
    pre_process(args_)
