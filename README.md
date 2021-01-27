<img align="left" src="https://github.com/nghitruyen/Text_Classification_Defi-IA-2021/blob/main/images/Logo_INSAvilletoulouse-RVB.png">
<br />
<br />
<br />
<br />

# Text Classification - Défi-IA 2021

## *Assigning the correct job category to a job description*

### Task overview:

We participed the 5th edition of the so-called Défi IA, which pertains to NLP (Natural Language Processing) and Text Classification. The task is to assign the correct job category to a job description. This is thus a multi-class classification task with 28 classes to choose from.

The data has been retrieved from [CommonCrawl](https://www.wikiwand.com/en/Common_Crawl). The latter has been famously used to train [OpenAI's GPT-3 model](https://www.wikiwand.com/en/GPT-3). One of the goals of this competition is to design a solution that is both accurate as well as fair.

### Tutorial:

#### About BERT:

[BERT](https://arxiv.org/abs/1810.04805) and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models. The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers. 

BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.

The code of our project allows to:

- Load the dataset,

- Load a BERT model from TensorFlow Hub,

- Build our own model by combining BERT with a classifier,

- Train our own model, fine-tuning BERT as part of that,

- Save our model and use it to classify sentences.

#### Files and folders organisation:

The main folders of this git are organized as below:

    ├── ...
    ├── data               
    │   ├── test    
    |   |   └── test.json
    │   ├── train
    |   |   ├── train_label.csv
    |   |   └── train.json
    │   └── categories_string.csv
    ├── output               
    │   ├── models      
    │   ├── results
    │   └── submission
    └── ...

The data folder contains the data on which we train and test the algorithm:

- `train.json`: contains job descriptions as well as genders for the training set, which contains 217,197 samples.

- `train_label.csv`: contains job labels for the training set.

- `categories_string.csv`: provides a mapping between job labels and integers, the latter of which are used for scoring submissions.

- `test.json`: contains job descriptions as well as genders for the test set, which contains 54,300 samples.

The `output` directory will be generated automatically when running our code:

- `models/`: contains the different models saved during the training.

- `results/`: contains the different results file produce during the training.

- `submission/`: contains the submission files. 

#### Code Execution:

Firstly, for setting the environment, we can check the version number of each package, which is presented in `requirements.txt`.

Then, the following command in terminal allows to run the learning and save the model as well as the result in `/output/models/` and `/output/results/`:

`python BERT_learning.py`

In order to run a prediction for the test set `test.json` and save it in `/output/submission/`, we run the command below (see *Current Issue* section for current bug in this script):

`python BERT_prediction.py`

Furthermore, the python scripts should be able to take arguments into account. For example, we can run the following commands:

`python BERT_learning.py --bootstrap True --batch_size 32 --epochs 5 --learning_rate 0.00005`

`python BERT_prediction.py --bootstrap True --batch_size 32 --epochs 5 --learning_rate 0.00005`

The arguments of our model here are:

- `bootstrap`: adding bootstrap training data in order to remove the unfairness across genders and classes.

- `batch_size`: size of the batch.

- `epochs`: number of epochs.

- `learning_rate`: learning rate in optimization algorithm.

### Results:

We achieved a public score of 0.80047 and a private score of 0.80789 which lead to the 28-th position and 20-th position on respectively public and private leaderboard of DefiIA2021. The details can be shown in [Kaggle](https://www.kaggle.com/c/defi-ia-insa-toulouse/).

This results is obtained by a Google colab runtime which utilized a Nvidia Tesla T4 as accelerator in a total runtime of 6 hours for original data and 7.5 hours for the bootstrapped data in our attempt of balancing the dataset. All the terminal command are executed through a notebook by using prefixes "!" or "%" before those commands, which should be equivalent to a terminal in a local runtime. 

Why didn't we use a local runtime? Our local runtime had a GTX 1050Ti 4GB as GPU accelerator which did not have enough VRAM to handle a BERT model. Indeed, every attempt to allocating resource for the model led to a dead python kernel. This problem would be solved once we get access to a more powerful local machine. Our solution was to emulate a terminal through a colab notebook, this workaround was necessary. 

### Existing Issues

The command `python BERT_prediction.py` returns `KeyError: 'RegexSplitWithOffsets'` (see `/notebook/terminal.ipynb` for more information) which, based on our research, is a tensorflow bug in recent versions. This problem only exists when `tensorflow.keras.models.save` and `tensorflow.keras.models.load` commands are executed in 2 separate scripts and `tensorflow_text` are being used. Therefore, we are waiting for newer version of tensorflow to fully resolve this bug.

In waiting, our temporary solution is to make a prediction on test dataset within the command `python BERT_learning.py`.

### Authors:

Ngo Nghi Truyen Huynh

Ngoc Bao Nguyen

Dinh Triem Phan

Viet Minh Thong Le

Adam Hamidallah

