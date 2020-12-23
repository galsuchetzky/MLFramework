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
import csv
import re
import pandas as pd

from args import get_setup_args
from codecs import open
from tqdm import tqdm
from zipfile import ZipFile
from nlp import load_dataset


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
    downloads = []
    if args.huggingface:
        load_dataset('break_data', 'QDMR', cache_dir='.\\data\\')
    else:
        downloads.append(('Break Dataset', args.dataset_url))

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
                print(output_path)
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


def get_example_split_set_from_id(question_id):
    return question_id.split('_')[1]


def fix_references(string):
    return re.sub(r'#([1-9][0-9]?)', '@@\g<1>@@', string)


def process_target(target):
    # replace multiple whitespaces with a single whitespace.
    target_new = ' '.join(target.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = target_new.split(';')
    new_parts = [re.sub(r'return', '', part.strip()) for part in parts]
    target_new = ' @@SEP@@ '.join([part.strip() for part in new_parts])

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    target_new = fix_references(target_new)

    return target_new.strip()


def preprocess_input_file(input_file, lexicon_file=None, model=None):
    if lexicon_file:
        lexicon = [
            json.loads(line)
            for line in open(lexicon_file, "r").readlines()
        ]
    else:
        lexicon = None

    examples = []
    with open(input_file, encoding='utf-8') as f:
        lines = csv.reader(f)
        header = next(lines, None)
        num_fields = len(header)
        assert num_fields == 5

        for i, line in enumerate(lines):
            # TODO: remove this, for debugging
            # if len(line) != 5:
            #     print("failed to read example:", i, "which is:", line.encode())
            #     continue
            assert len(line) == num_fields, "read {} fields, and not {}".format(len(line), num_fields)
            question_id, source, target, _, split = line
            split = get_example_split_set_from_id(question_id)

            target = process_target(target)
            example = {'annotation_id': '', 'question_id': question_id,
                       'source': source, 'target': target, 'split': split}
            if model:
                parsed = model(source)
                example['source_parsed'] = parsed
            if lexicon:
                # TODO: remove this, for debugging.
                # print(example['source'].encode())
                # print(lexicon[i]['source'].encode())
                # if not example['source'] == lexicon[i]['source']:
                #     print("failed in lexicon comparison for:", lexicon[i]['source'], "in the lexicon file.")
                assert example['source'] == lexicon[i]['source']
                example['allowed_tokens'] = lexicon[i]['allowed_tokens']

            examples.append(example)

    return examples


def pre_process(args):
    """
    Preprocess your dataset.
    Args:
        args: all arguments recived by the user as well ad the default ones.
    """
    # TODO: add your preprocess code here.
    OUTPUT_DIR = 'data\\'
    QDMR_BASE_DIR = 'data\\Break-dataset\\QDMR\\'
    files = [(QDMR_BASE_DIR, 'train.csv', 'train_lexicon_tokens.json', 'QDMR_train'),
             (QDMR_BASE_DIR, 'dev.csv', 'dev_lexicon_tokens.json', 'QDMR_dev'),
             (QDMR_BASE_DIR, 'test.csv', 'test_lexicon_tokens.json', 'QDMR_test')
             ]

    for base_dir, csv_file, lexicon, output_file_base in files:
        examples = preprocess_input_file(base_dir + csv_file, base_dir + lexicon)
        print(f"processed {len(examples)} examples.")
        if args.sample:
            examples = sample_examples(examples, args.sample)
            print(f"left with {len(examples)} examples after sampling.")

        # dynamic_vocab = args.lexicon_file is not None
        # write_output_files(os.path.join(args.output_dir, args.output_file_base), examples, dynamic_vocab)
        write_output_files(os.path.join(OUTPUT_DIR, output_file_base), examples, True)

    print("done!\n")

    # Note: after preprocessing, you can save the preprocessed results in a json file using the save function.
    # example:
    # save(args.word_emb_file, word_emb_mat, message="word embedding")


def write_output_files(base_path, examples, dynamic_vocab):
    # Output file is suitable for the allennlp seq2seq reader and predictor.
    with open(base_path + '.tsv', 'w', encoding='utf-8') as fd:
        for example in examples:
            if dynamic_vocab:
                output = example['source'] + '\t' + example['allowed_tokens'] + '\t' + example['target'] + '\n'
            else:
                output = example['source'] + '\t' + example['target'] + '\n'
            fd.write(output)

    with open(base_path + '.json', 'w', encoding='utf-8') as fd:
        for example in examples:
            output_dict = {'source': example['source']}
            if dynamic_vocab:
                output_dict['allowed_tokens'] = example['allowed_tokens']
            fd.write(json.dumps(output_dict) + '\n')

    print(base_path + '.tsv')
    print(base_path + '.json')


def sample_examples(examples, configuration):
    df = pd.DataFrame(examples)
    df["dataset"] = df.question_id.apply(lambda x: x.split('_')[0])

    print("dataset distribution before sampling:")
    print(df.groupby("dataset").agg("count"))
    for dataset in df.dataset.unique().tolist():
        if dataset in configuration:
            drop_frac = 1 - configuration[dataset]
            df = df.drop(df[df.dataset == dataset].sample(frac=drop_frac).index)

    print("dataset distribution after sampling:")
    print(df.groupby("dataset").agg("count"))

    return df.to_dict(orient="records")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Preprocess dataset
    # TODO: add any required paths for preprocess to args_.

    # if args_.include_test_examples:  # If should include the test examples in the preprocess.
    #     args_.test_file = url_to_data_path(args_.test_url)

    # TODO: add to args_ anything else required for the preprocessing of your data.
    pre_process(args_)
