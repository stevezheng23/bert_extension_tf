# BERT Extension
BERT (Bidirectional Encoder Representations from Transformers) is a generalized autoencoding pretraining method proposed by Google AI Language team, which obtains new state-of-the-art results
on 11 NLP tasks ranging from question answering, natural, language inference and sentiment analysis. BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers, which allows it to be easily finetuned for downstream tasks without substantial taskspecific architecture modifications. This project is aiming to provide extensions built on top of current BERT and bring power of BERT to other NLP tasks like NER and NLU.
<p align="center"><img src="/docs/bert.tasks.png" width=800></p>
<p align="center"><i>Figure 1: Illustrations of fine-tuning BERT on different tasks</i></p>

## Setting
* Python 3.6.7
* Tensorflow 1.13.1
* NumPy 1.13.3

## DataSet
* [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/) is a multi-task dataset, which contains 3 sub-tasks, POS tagging, syntactic chunking and NER. For NER sub-task, it contains 4 types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* [ATIS](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html) is NLU dataset in airline travel domain. The dataset contains 4978 train and 893 test utterances classified into one of 26 intents, and each token in utterance is labeled with tags from 128 slot filling tags in IOB format.

## Reference
* Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee,
and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
