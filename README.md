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

For setting the environment, we can check the version number of each package, which is presented in `requirements.txt`.

`python BERT_learning.py`

`python BERT_learning.py --bootstrap True --batch_size 32 --epochs 5 --learning_rate 0.00005`

`python BERT_prediction.py`

`python BERT_prediction.py --bootstrap True --batch_size 32 --epochs 5 --learning_rate 0.00005`

### Results:

### Authors:

Ngo Nghi Truyen Huynh

Ngoc Bao Nguyen

Dinh Triem Phan

Viet Minh Thong Le

Adam Hamidallah

