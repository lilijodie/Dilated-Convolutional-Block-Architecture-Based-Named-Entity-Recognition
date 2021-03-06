# Dilated-Convolutional-Block-Architecture-Based-Named-Entity-Recognition

This repository includes the code for building __Dilated Convolutional Block Architecture Based Named Entity Recognition Model__ for Chinese News Named Entity Recognition task. Its goal is to recognize three types of Named Entity: PERSON, LOCATION and ORGANIZATION.

This code works on __Python 3.6.5 & TensorFlow 1.8__ and the following repositories [https://github.com/Acstream/Dilated-Block-Based-Convolutional-Network] and [https://github.com/Determined22/zh-NER-TF] gives me much help.

For more details, please view our paper __"Named Entity Recognition Model based on Dilated Convolutional Block Architecture"(CCML 2021)__ By Yue Yuan, Yanli Wang and Kan Liu (URL: https://pan.baidu.com/s/123bgbNT1kBgA8mgwbs_nOw Extraction Code: giqv).

## Model
You can build three kinds of models (including two of our proposed models) in __model.py__ just by annotating some codes in "def build_graph():" function (self.dcba_op() and self.dcl_op()):

For building __DCBA+Bi-LSTM+CRF__ model (_our proposed model_), you need to annotate "self.dcl_op()" and keep "self.dcba_op()" and other codes in "def build_graph():" remain.

For building __DCL+Bi-LSTM+CRF__ model (_our proposed model_), you need to annotate "self.dcba_op()" and keep "self.dcl_op()" and other codes in "def build_graph():" remain.

For building __Bi-LSTM+CRF__ model, you need to annotate "self.dcba_op()" together with "self.dcl_op()" and keep other codes in "def build_graph():" remain.

The above models are proposed by our paper __"Named Entity Recognition Model based on Dilated Convolutional Block Architecture"(CCML 2021)__. 

The structure of __"Dilated Convolutional Block Architecture Based Named Entity Recognition Model"__(DCBA+Bi-LSTM+CRF) looks like the following illustration:

![DCBA+Bi-LSTM+CRF](./pic1.png)

The structure of __"Dilated Convolutional Layer Based Named Entity Recognition Model"__(DCL+Bi-LSTM+CRF) looks like the following illustration:

![DCL+Bi-LSTM+CRF](./pic2.png)

For one Chinese sentence, each character in this sentence has / will have a tag which belongs to the set {O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}.

The first layer, __look-up layer__, aims at transforming each character representation from one-hot vector into *character embedding*.???

The second layer, __DCBA or DCL layer__, can efficiently extract contextual features from input embeddings.

The third layer, __BiLSTM layer__, can efficiently use *both past and future* input information and extract features automatically.

The fourth layer, __CRF layer__,  labels the tag for each character in one sentence. If we use a Softmax layer for labeling, we might get ungrammatic tag sequences beacuse the Softmax layer labels each position independently. We know that 'I-LOC' cannot follow 'B-PER' but Softmax doesn't know. Compared to Softmax, a CRF layer can use *sentence-level tag information* and model the transition behavior of each two different tags.

## Dataset

The dataset is a portion of [MSRA corpus](http://sighan.cs.uchicago.edu/bakeoff2006/). You can download the original dataset from the link in `./data_path/link.txt`

### data files

The directory `./data_path` contains:

- the preprocessed data files, `train_data` and `test_data` 
- a vocabulary file `word2id.pkl` that maps each character to a unique id  
- a link file `link.txt` contains a url for downloading the original dataset  

For generating vocabulary file, please refer to the code in `data.py`. 

### data format

Each data file should be in the following format:

```
???	B-LOC
???	I-LOC
???	O
???	O

???	O
???	O
???	O
???	O
???	O
???	O
???	O

```

If you want to use your own dataset, please: 

- transform your corpus to the above format
- generate a new vocabulary file

## How to Run

### train

`python main.py --mode=train `

### test

To choose the test our trained models (experiment) please refer to the directory "data_path_save".

`python main.py --mode=test --demo_model=dcba+bilstm+crf`

Please set the parameter `--demo_model` to the model that you want to test. `dcba+bilstm+crf` is one of the models trained by me. 

Those trained models are: dcba+bilstm+crf, dcl+bilstm+crf, bilstm+crf, dcba+bilstm+softmax, dcl+bilstm+softmax, bilstm+softmax.

For idcnn+crf model that mentioned in our paper, please refer to [https://github.com/kungfulei/NER_BiLSTM_IDCNN_CRF].

An official evaluation tool for computing metrics: [here (click 'Instructions')](http://sighan.cs.uchicago.edu/bakeoff2006/)

__Please do remember to install Perl on your machine before evaluation!__


