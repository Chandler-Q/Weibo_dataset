# Weibo_dataset
This is a Python script for processing raw weibo data to build a PyG dataset. This repo includes the Pytorch-Geometric implementation of our Weibo dataset.

We referenced the work of [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) and [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) in creating this dataset.
## Instruction
We constructed a Weibo dataset, the original data of the dataset comes from the Chinese Weibo social platform, including a total of 4,664 news, each news constitutes a network, which is represented as a graph, the root node of each tree is "news", and all the other leaf nodes represent "users", as shown in the following figure:

![Graph Representation](https://github.com/Chandler-Q/GANM/graph.jpg)

We used pre-trained embeddings [spaCy](https://spacy.io/models/zh#zh_core_web_lg) word2vec and [BERT](https://github.com/jina-ai/clip-as-service) for embedding process.
|  Weibo   | #Total Graphs  | #Fake News  |#Total Nodes|#Total Edges| Profile size  | word2vec size  | BERT size|
|  ----    | ----           | ----        |      ----  | ----       | ----  | ----  | ----  |
|          |  4,664         | 2313        |2,856,741   |2,852,077    | 14   |314| 768|



## Files Note
|  File   | Note  |
|  ----  | ----  |
| data  | raw: raw data; Weibo: raw embeddings and processed dataset|
|  Graph_processing.py | The script for processing graphs Some pre-processing work was done with it.|
|  encoding.py | word2vec and BERT pre-trained embeddings |
|  dataset_processing.py | The main runtime script that performs the raw data embedding process. |
|  Weibo.py | Post-embedding data (in . \data\raw) generated from dataset_processing.py  to generate a my PyG dataset. |

