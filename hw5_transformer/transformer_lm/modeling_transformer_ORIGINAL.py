#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.modeling_attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0, causal=False):
        super().__init__()
        # Task 2.1 (1 point)
        # Create layers needed for Transformer Encoder Block
        # (5 layers in total, including dropout)
        # We recommend to use nn.Sequential for FFN instead of creating is layer-by-layer,
        # it will make your code more readable
        # YOUR CODE STARTS HERE  (our implementation is about 5-8 lines)    

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """Self-Attention -> residual -> LayerNorm -> FCN -> residual -> LayerNorm
        
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """

        # Task 2.2 (2 points)
        # Implement Transformer encoder block forward pass
        # You can implement residual connection this way:
        # residual = x
        # x = some_stuff_that_changes_x(x)
        # x = x + residual
        # YOUR CODE STARTS HERE (our implementation is about 6 lines)

        # YOUR CODE ENDS HERE
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1, causal=False):
        """A minimal implementation of Transformer Encoder
        
        Args:
            num_layer: number of encoder layer
            hidden: embedding size and hidden size of attentions
            fcn_hidden: hidden size of fully-connected networks inside transformer layers
            vocab_size: size of vocabulary
            max_seq_len: maximum length of input sequence
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.fcn_hidden = fcn_hidden
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        # Task 2.3 (2 points)
        # 1. Create embedding layer and positional embedding layer
        # Use nn.Embedding for that
        # 2. Create a linear layer logit_proj that will project contextualized representations
        # of size hidden to your vocabulary size.
        # 3. Create a dropout layer
        # 4. Create a list of encoder Layers
        # Note that you need to wrap it with nn.ModuleList,
        # so that the parameters of the layers would be counted as the paramertes of the model
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # Read more about ModuleList here:
        # https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
        # You can use for-loop of python list comprehension to create the list of layers
        # YOUR CODE STARTS HERE (our implementation is about 6 lines)

        # YOUR CODE ENDS HERE

    def _add_positions(self, sequence_tensor):
        """Adds positional embeddings to the input tensor.

        Args:
            sequence_tensor: FloatTensor[batch_size, seq_len, hidden]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        seq_len = sequence_tensor.shape[1]

        # Task 2.5 (1 point)
        # Implement positional embedding which is a sum of:
        # 1. Embedding of the position (self.pos_emb)
        # 2. Embedding of the token (sequence_tensor)
        # Remember that is you create any tensors here,
        # you need to move them to the same device as sequence_tensor
        # You can get device of sequence_tensor with sequence_tensor.device
        # YOUR CODE STARTS HERE (our implementation is about 3 lines)

        # YOUR CODE ENDS HERE

    def forward(self, input_ids=None):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len]
        
        Returns:
            FloatTensor[batch_size, src_seq_len, hidden]
        """
        # Task 2.6 (2 points)
        # Implement Transformer Encoder
        # Remember that Transformer Encoder is composed of:
        # 1. Embedding
        # 2. Positional Embedding (use self._add_positions)
        # 3. Transformer Encoder Layers
        # NOTE: Please write shape of the tensor for each line of code
        # YOUR CODE STARTS HERE (our implementation is about 4 lines)

        # YOUR CODE ENDS HERE


class TransformerLM(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1):
        """Transformer Language Model"""
        super().__init__()
        self.dropout_rate = dropout

        # Task 2.7 (1 point)
        # Create a Transformer Encoder, output layer for language modeling, and a dropout layer
        # Remember that when we use Transformer for language modeling, it should be **causal** or it will cheat.
        # Output layer should predict the logits for all words in the vocabulary (size of logits = vocab_size)
        # YOUR CODE STARTS HERE (our implementation is about 2 lines)

        # YOUR CODE ENDS HERE
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len], optional, encoder_embeds could be used instead
        Returns:
            FloatTensor[batch_size, src_seq_len, vocab_size] — logits over the vocabulary
        """
        assert input_ids.dim() == 2, "Input should be of size [batch_size, seq_len]"
        # Task 2.8 (1 point)
        # Implement Transformer Language Model
        # Remember that Transformer Language Model is composed of:
        # 1. Transformer Encoder
        # 2. Dropout
        # 3. Output Layer to produce logits over the classes (our vocabulary in case of language modeling)
        # YOUR CODE STARTS HERE (our implementation is 2 lines)

        # YOUR CODE ENDS HERE

    def save_pretrained(self, save_path):
        """Save the model weights to a directory

        Args:
            save_path: directory to save the model
        """
        config = {
            "num_layers": self.encoder.num_layers,
            "hidden": self.encoder.hidden,
            "num_heads": self.encoder.num_heads,
            "fcn_hidden": self.encoder.fcn_hidden,
            "vocab_size": self.encoder.vocab_size,
            "max_seq_len": self.encoder.max_seq_len,
            "dropout": self.encoder.dropout_rate,
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
           json.dump(config, f)

        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, save_path):
        """Load the model weights from a directory

        Args:
            save_path: directory to load the model
        """
        with open(os.path.join(save_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_path, "model.pt"), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model
