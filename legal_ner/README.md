# Legal Named Entity Recognition
This folder contains the code and the data for the Legal NER task (Task 6.B).


## Requirements
The code is written in Python 3.10. The required packages are listed in `requirements.txt`. To install the required packages, run:

    pip install -r requirements.txt

## Code 
The main code for the L-NER task allowing to fine-tune the models is available in the `main.py` script.  
The `inference.py` script allows instead to predict the labels for the test set.

The `utils` folder contains the code for the data loading and the evaluation.

The data are not included in this repository as they are not yet publicly available.
More information are provided in the [official SemEval-2023 Task 6 website](https://sites.google.com/view/legaleval/home).


## Usage
To fine-tune the models on the train data and evaluate them on the dev set, run:

    python main.py

To predict the labels for the test set, run:

    python inference.py
