# Training Strategy for Kosmos-X
This document outlines the training strategy, architecture, datasets to prioritize, metrics to expect, and best practices to follow for the Kosmos-X Transformer model.

## 1. Architecture
Kosmos-X uses a decoder-only Transformer architecture based on Magneto (Foundation Transformers). The architecture employs a sub-LN approach, where layer normalization is added both before the attention module (pre-ln) and afterwards (post-ln). This combines the advantages of language modeling and image understanding.

Images are encoded into image features using a CLIP VIT-L/14 model, and a perceiver resampler introduced in Flamingo is used to pool the image features from 256 to 64 tokens. The image features are combined with the token embeddings by adding them to the input sequence surrounded by special tokens <image> and </image>. An example is <s> <image> image_features </image> text </s>. This allows images to be interwoven with text in the same sequence.

## 2. Datasets
2.1 Pretraining
2.1.1 Text Corpora
Kosmos-X is trained on The Pile and Common Crawl. The Pile is an 800 GB English text corpus combining 22 diverse sources. A subset with seven sources from The Pile is selected. Common Crawl takes snapshots of the web, which contains massive amounts of language data. The language datasets used in the training of Kosmos-X can be divided into the following three categories:

Prioritized Datasets for Training Kosmos-X
The following list of prioritized datasets is recommended for training Kosmos-X, with metrics such as recommended batch size, epochs, tokens, and weight:

High Priority
OpenWebText2

Tokens (billion): 14.8
Weight (%): 21.8%
Epochs: 1.47
CC-2021-04

Tokens (billion): 82.6
Weight (%): 17.7%
Epochs: 0.21
Books3

Tokens (billion): 25.7
Weight (%): 16.2%
Epochs: 0.63
Medium Priority
CC-2020-50


Tokens (billions): 11M images and 1.1B mask annotations
Weight: (%): 10%
Epochs: 0.51
SA-1B
[SA-1B](https://ai.facebook.com/datasets/segment-anything-downloads/)

Tokens (billion): 68.7
Weight (%): 14.7%
Epochs: 0.21
Pile-CC

Tokens (billion): 49.8
Weight (%): 10.6%
Epochs: 0.21
Realnews

Tokens (billion): 21.9
Weight (%): 10.2%
Epochs: 0.46
Wikipedia (English)

Tokens (billion): 4.2
Weight (%): 5.4%
Epochs: 1.29
Low Priority
BookCorpus2

Tokens (billion): 1.5
Weight (%): 1.1%
Epochs: 0.75
Gutenberg (PG-19)

Tokens (billion): 2.7
Weight (%): 1.0%
Epochs: 0.38
CC-Stories

Tokens (billion): 5.3
Weight (%): 1.0%
Epochs: 0.19
NIH ExPorter

Tokens (billion): 0.3
Weight (%): 0.2%
Epochs: 0.75
These datasets can be further categorized into:

Academic: NIH Exporter
Internet: Pile-CC, OpenWebText2, Wikipedia (English), CC-2020-50, CC-2021-04, Realnews
Prose: BookCorpus2, Books3, Gutenberg, CC-Stories
Additionally, consider incorporating the following datasets to enhance the model's multimodal capabilities:

ImageNet
COCO
CUB-200-2011
SST
Flickr30k
SuperGLUE
Visual Question Answering v2.0
Conceptual Captions
BoolQ
COPA
COCO Captions
HellaSwag
WinoGrande
PIQA
The Pile
Hateful Memes
VizWiz

Academic: NIH Exporter
Internet: Pile-CC, OpenWebText2, Wikipedia (English), CC-2020-50, CC-2021-04, Realnews
Prose: BookCorpus2, Books3, Gutenberg, CC-Stories

2.1.2 Image-Caption Pairs
Kosmos-X is trained on image-caption pairs constructed from several datasets, including English LAION-2B, LAION-400M, COYO-700M, and Conceptual Captions. These datasets are extracted by parsing out image URLs and alt-texts of web pages from the Common Crawl web data.

## 2.1.3 Interleaved Data
A large corpus of 2 billion web pages is collected from the snapshots of common crawls. Several filtering criteria are applied to ensure quality and relevance. After applying these filters, about 71 million documents are used for training.

## 2.2 Data Format
The training data is organized in the following format:

Text: <s> KOSMOS-1 can perceive multimodal input, learn in context, and generate output. </s>
Image-Caption: <s> <image> Image Embedding </image> WALL-E giving potted plant to EVE. </s>
Multimodal: <s> <image> Image Embedding </image> This is WALL-E. <image> Image Embedding </image> This is EVE. </s>
3. Metrics to Expect
The performance of Kosmos-X can be evaluated using various metrics, such as:

Perplexity: A measure of how well the model predicts the test data. Lower perplexity indicates better performance.
BLEU score: A metric for evaluating the quality of generated text by comparing it to reference translations. Higher BLEU scores indicate better performance.
F1 score: A measure of the model's accuracy in predicting specific tasks, such as question-answering or summarization. Higher F1 scores indicate better performance.
4. Best Practices
Use a diverse set of high-quality datasets for pretraining to ensure the model learns a wide range of language and image understanding tasks.
Regularly monitor the training progress and adjust hyperparameters, such as learning rate and batch size, to optimize the model's performance.
Employ data augmentation techniques, such as random cropping and flipping of images, to increase the diversity of the training data and improve the model's generalization capabilities.
Fine-tune the model on task-specific datasets to achieve better performance on specific tasks, such as image captioning or text summarization.