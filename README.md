# BERT Extension
[BERT](https://github.com/google-research/bert/) (Bidirectional Encoder Representations from Transformers) is a generalized autoencoding pretraining method proposed by Google AI Language team, which obtains new state-of-the-art results on 11 NLP tasks ranging from question answering, natural, language inference and sentiment analysis. BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers, which allows it to be easily finetuned for downstream tasks without substantial task-specific architecture modifications. This project is aiming to provide extensions built on top of current BERT and bring power of BERT to other NLP tasks like NER and NLU.
<p align="center"><img src="/docs/bert.tasks.png" width=800></p>
<p align="center"><i>Figure 1: Illustrations of fine-tuning BERT on different tasks</i></p>

## Setting
* Python 3.6.7
* Tensorflow 1.13.1
* NumPy 1.13.3

## DataSet
* [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/) is a multi-task dataset, which contains 3 sub-tasks, POS tagging, syntactic chunking and NER. For NER sub-task, it contains 4 types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* [ATIS](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html) (Airline Travel Information System) is NLU dataset in airline travel domain. The dataset contains 4978 train and 893 test utterances classified into one of 26 intents, and each token in utterance is labeled with tags from 128 slot filling tags in IOB format.

## Usage
* Preprocess data
```bash
python prepro/prepro_conll.py \
  --data_format json \
  --input_file data/ner/conll2003/raw/eng.xxx \
  --output_file data/ner/conll2003/xxx-conll2003/xxx-conll2003.json
```
* Run experiment
```bash
CUDA_VISIBLE_DEVICES=0 python run_ner.py \
    --task_name=conll2003 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --do_export=true \
    --data_dir=data/ner/conll2003 \
    --vocab_file=model/cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=model/cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=model/cased_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --eval_batch_size=8 \
    --predict_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=5.0 \
    --output_dir=output/ner/conll2003/debug
    --export_dir=output/ner/conll2003/export
```
* Visualize summary
```bash
tensorboard --logdir=output/ner/conll2003
```
* Setup service
```bash
docker run -p 8500:8500 \
  -v output/ner/conll2003/export/xxxxx:models/ner \
  -e MODEL_NAME=ner \
  -t tensorflow/serving
```

## Experiment
### CoNLL2003-NER
<p align="center"><img src="/docs/bert.ner.png" width=500></p>
<p align="center"><i>Figure 2: Illustrations of fine-tuning BERT on NER task</i></p>

|    CoNLL2003 - NER  |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|      Precision      |   91.37 ± 0.33   |     91.87     |
|         Recall      |   92.37 ± 0.25   |     92.68     |
|       F1 Score      |   91.87 ± 0.28   |     92.27     |

<p><i>Table 1: The test set performance of BERT-large finetuned model on CoNLL2003-NER task with setting: batch size = 16, max length = 128, learning rate = 2e-5, num epoch = 5.0</i></p>

### ATIS-NLU
<p align="center"><img src="/docs/bert.nlu.png" width=500></p>
<p align="center"><i>Figure 3: Illustrations of fine-tuning BERT on NLU task</i></p>

|      ATIS - NLU     |   Avg. (5-run)   |      Best     |
|:-------------------:|:----------------:|:-------------:|
|  Accuracy - Intent  |   97.38 ± 0.19   |     97.65     |
|    F1 Score - Slot  |   95.61 ± 0.09   |     95.53     |

<p><i>Table 2: The test set performance of BERT-large finetuned model on ATIS-NLU task with setting: batch size = 16, max length = 128, learning rate = 2e-5, num epoch = 5.0</i></p>

## Reference
* Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee,
and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le. [XLNet: Generalized autoregressive pretraining for language understanding](https://arxiv.org/abs/1906.08237) [2019]
