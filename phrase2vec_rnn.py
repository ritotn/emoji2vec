#!/usr/bin/env python
"""Wrapper for word2vec and emoji2vec models, so that we can query by entire phrase, rather than by
individual words.
"""

# External dependencies
import os.path
import gensim.models as gs
import numpy as np

from typing import List, Dict, Union, Tuple
import sklearn 
from sklearn import linear_model
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def add_the_embedding(embed_array, vocab2indx): 
    """
    Adds "the" embedding to the embed_array matrix
    """
    the_embedding = embed_array[vocab2indx["the"]]
    out = np.vstack((embed_array, the_embedding))
    return out

def create_word_indices(tokens: List[str], vocab2indx: dict) -> List[int]: 
    """
    For each example, translate each token into its corresponding index from vocab2indx
    
    Replace words not in the vocabulary with the symbol "<OOV>" 
        which stands for 'out of vocabulary'
        
    Arguments: 
       - tokens (List[str]): list of strings of tokens 
       - vocab2indx (dict): each vocabulary word as strings and its corresponding int index 
                           for the embeddings 
                           
    Returns: 
        - (List[int]): list of integers
    """ 
    return [vocab2indx[token] if token in vocab2indx else vocab2indx["<OOV>"] for token in tokens]

def truncate(original_indices_list: list, maximum_length=300) -> list: 
    """
    Truncates the original_indices_list to the maximum_length
    """
    return original_indices_list[0:maximum_length]

def pad(original_indices_list: list, vocab2indx, maximum_length=300) -> list: 
    """
    Given original_indices_list, concatenates the pad_index enough times 
    to make the list to maximum_length. 
    """
    return original_indices_list + [vocab2indx["<PAD>"] for i in range(maximum_length - len(original_indices_list))]

def convert_to_indices(tokens, vocab2indx):
    MAXIMUM_LENGTH = 300
    
    token_indices = create_word_indices(tokens, vocab2indx)
    token_indices = truncate(token_indices, maximum_length=MAXIMUM_LENGTH)
    token_indices = pad(token_indices, vocab2indx, maximum_length=MAXIMUM_LENGTH)

    return torch.LongTensor(token_indices)


class Phrase2VecRNN(nn.Module):
    def __init__(self, dim, w2v_path, e2v_path=None, hidden_dim=300, num_layers=1, dropout_prob=0.1):
        """Constructor for the Phrase2Vec model

        Args:
            dim: Dimension of the vectors in word2vec and emoji2vec
            w2v: Gensim object for word2vec
            e2v: Gensim object for emoji2vec
        """
        super().__init__()
        if not os.path.exists(w2v_path):
            print(str.format('{} not found. Either provide a different path, or download binary from '
                                'https://code.google.com/archive/p/word2vec/ and unzip', w2v_path))

        self.wordVecModel = gs.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

        if e2v_path is not None:
            self.emojiVecModel = gs.KeyedVectors.load_word2vec_format(e2v_path, binary=True)
        else:
            self.emojiVecModel = dict()

        vocab2indx = dict(self.wordVecModel.key_to_index)
        idx2vocab = list(self.wordVecModel.index_to_key)
        embed_array = self.wordVecModel.vectors
        
        # add <OOV> symbol
        new_oov_entry = len(self.wordVecModel)
        idx2vocab += ["<OOV>"]
        vocab2indx["<OOV>"] = new_oov_entry
        embed_array_w_oov = add_the_embedding(embed_array, vocab2indx)

        # Add <PAD> symbol (also as embedding for the word type "the")
        new_pad_entry = len(idx2vocab)
        idx2vocab += ["<PAD>"]
        vocab2indx["<PAD>"] = new_pad_entry

        self.vocab2indx = vocab2indx
        self.idx2vocab = idx2vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        embed_array_w_oov_pad = add_the_embedding(embed_array_w_oov, vocab2indx)
        vecs = torch.FloatTensor(embed_array_w_oov_pad)

        self.embeddings = nn.Embedding.from_pretrained(vecs, freeze=True)
        self.dropout = nn.Dropout(dropout_prob) 
        self.linear = nn.Linear(hidden_dim, 300)

        self.rnn = nn.RNN(input_size=dim, hidden_size=300, num_layers=num_layers)  

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        embed = self.embeddings(x)
        out, hidden = self.rnn(embed, hidden)
        out = self.dropout(out)
        out = self.linear(out)

        return out, hidden

    def __getitem__(self, item):
        tokens = item.split(' ')
        token_indices = convert_to_indices(tokens, self.vocab2indx)
        out, _ = self.forward(token_indices)
        return out
        
    def from_emoji(self, emoji_vec, top_n=10):
        """Get the top n closest tokens for a supplied emoji vector

        Args:
            emoji_vec: Emoji vector
            top_n: number of results to return

        Returns:
            Closest n tokens for a supplied emoji_vec
        """
        return self.wordVecModel.most_similar(positive=emoji_vec, negative=[], topn=top_n)

    def __setitem__(self, key, value):
        self.wordVecModel[key] = value


class Phrase2VecDAN(nn.Module):
    def __init__(self, dim, w2v_path, e2v_path=None, hidden_dim1=300, hidden_dim2=300, output_dim=300, leaky_relu_slope=0.01, dropout_prob=0.3):
        """Constructor for the Phrase2Vec model

        Args:
            dim: Dimension of the vectors in word2vec and emoji2vec
            w2v: Gensim object for word2vec
            e2v: Gensim object for emoji2vec
        """
        super().__init__()
        if not os.path.exists(w2v_path):
            print(str.format('{} not found. Either provide a different path, or download binary from '
                                'https://code.google.com/archive/p/word2vec/ and unzip', w2v_path))

        self.wordVecModel = gs.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

        if e2v_path is not None:
            self.emojiVecModel = gs.KeyedVectors.load_word2vec_format(e2v_path, binary=True)
        else:
            self.emojiVecModel = dict()

        vocab2indx = dict(self.wordVecModel.key_to_index)
        idx2vocab = list(self.wordVecModel.index_to_key)
        embed_array = self.wordVecModel.vectors
        
        # add <OOV> symbol
        new_oov_entry = len(self.wordVecModel)
        idx2vocab += ["<OOV>"]
        vocab2indx["<OOV>"] = new_oov_entry
        embed_array_w_oov = add_the_embedding(embed_array, vocab2indx)

        # Add <PAD> symbol (also as embedding for the word type "the")
        new_pad_entry = len(idx2vocab)
        idx2vocab += ["<PAD>"]
        vocab2indx["<PAD>"] = new_pad_entry

        self.vocab2indx = vocab2indx
        self.idx2vocab = idx2vocab

        embed_array_w_oov_pad = add_the_embedding(embed_array_w_oov, vocab2indx)
        vecs = torch.FloatTensor(embed_array_w_oov_pad)

        self.embeddings = nn.Embedding.from_pretrained(vecs, freeze=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)

        embed = self.embeddings(x)
        out = torch.mean(embed, 1)
        out = self.dropout(out)
        out = self.model(out)

        return out

    def __getitem__(self, item):
        tokens = item.split(' ')
        token_indices = convert_to_indices(tokens, self.vocab2indx)
        out = self.forward(token_indices)
        return out
        
    def from_emoji(self, emoji_vec, top_n=10):
        """Get the top n closest tokens for a supplied emoji vector

        Args:
            emoji_vec: Emoji vector
            top_n: number of results to return

        Returns:
            Closest n tokens for a supplied emoji_vec
        """
        return self.wordVecModel.most_similar(positive=emoji_vec, negative=[], topn=top_n)

    def __setitem__(self, key, value):
        self.wordVecModel[key] = value

