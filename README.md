## description
This is the accompanying repository to Comparing Methods for Generating emoji2vec Embeddings, an extension of [emoji2vec: Learning Emoji 
Representations from their Description](https://arxiv.org/pdf/1609.08359.pdf) (Eisner et al. 2016). See the original repository [here](https://github.com/uclnlp/emoji2vec). 

Our edits involve adding additional methods to generate the emoj2vec embeddings. Like the original paper, we trained on descriptions from the Unicode standard, using the Google News word2vec embeddings. However, we created three new models, which are defined in `phrase2vec.py` and `phrase2vec_rnn.py`. These models include:
- vector averaging,
- a recurrent neural network (RNN)
- and a deep averaging network (DAN). 

## training
To train the embeddings using these models, we defined an additional hyperparameter in `parameter_parser.py`:

`-mod`: the model we use to train the emoji2vec embeddings

This can be any of the models:
- `‘sum’`: vector summation (original paper’s method)
- `‘avg’`: vector averaging (default)
- `‘rnn’`: recurrent neural network
- `‘dan’`: deep averaging network

We updated the code in `utils.py` accordingly to support generating embeddings using our models. 
