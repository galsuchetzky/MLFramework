# ML template
This project is a template project for a general deep learning project. <br>
It provides a base architecture for the project as well as some useful functions and code, <br>
such as:
- Logging
- Base environment
- Base training loop structure
- etc...

In each file, you will find documentation for the code and templates already existing in the file, <br>
as well as comments, TODOs and recommendations for what logic might be implemented in that part.

Note that the provided code is pytorch based, but can be translated into tensorflow if necessary.

## Files Description
### environment.yml
This file is a yml file for setting up an environment using the anaconda package manager.<br>
In it, you are already provided with a template for the file's structure, and some dependencies.<br>
Change this file and add all your dependencies as you go along with your project to keep the setup of a new 
environment simple.

Creating the environment:
1) Run the anaconda command line.
2) run `conda env create -f environment.yml`

now you have a new conda environment with the name specified in the yml file.<br>
In the case of the template environment, the name of the new environment is MLFramework. 

Activating the environment:
Each time you would like to work in the environment (via terminal or editors that do not remember the environment) 
you should run `activate MLFramework` to activate the environment.


todo: add anaconda installation details, basic commands, package installation details.

### Defaults.py
This file contains the list of default values, as well as all the project constants. <br>

Usage:<br>
- Add to that file all the default values for the command line arguments (see Args.py below) and any project constants.

### Args.py
This file is defining all the command line arguments for the all scripts in the project.
For example, the command line arguments that specify the sources for the dataset, the learning rate and the data 
paths. <br>
The file contains basic arguments, and many more examples for possible arguments, and you may change these according 
to need.
The argument parsing is based on the argparse library.

Usage: <br>
In the file itself you will find examples for arguments as well as documentation and explanation on adding new 
arguments.


### Config.py
This file contains the configuration classes for the project.<br>
Each config class takes care of parsing the command-line arguments that are relevant to it.
Currently, there are 3 config classes it this file: SetupConfig, TrainConfig, TestConfig.

Usage:
- If necessary, add more config classes.
- Edit the SetupConfig, TrainConfig and TestConfig classes to parse all the arguments relevant to this class.
- Make sure to include processing for any additional arguments that you have added to the Args.py file.

### Setup.py
This file is the setup script for the project. <br>
Here, all the required resources for the project will be downloaded, including the dataset.
Additionally, the preprocessing of the dataset will occur here as well.<br>
Note that this script will only run once.

Usage: <br>
- Edit the Args.py to include arguments to receive the resources parameters that are needed.
- Edit the download function to correctly download the resources (more details in the function).
- Edit the pre_process function to perform preprocessing to your downloaded dataset.


### Models.py
todo: add explanation and usage details

### Layers.py
todo: add explanation and usage details

### Trainers.py
todo: add explanation and usage details

### Train.py
todo: add explanation and usage details

### Test.py
todo: add explanation and usage details

### Utils.py
todo: add explanation and usage details


## Setup

1. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `MLFramework`

3. Run `source activate MLFramework`
    1. This activates the `MLFramework` environment
    2. Do this each time you want to write/test your code
<!--  
4. Run `python Setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `Setup.py` takes around 30 minutes total  

5. Browse the code in `Train.py`
    1. The `Train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python Train.py -h`.
   -->
