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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden):
        """Self-attention module which computes softmax(xQ @ xK^T) @ xV

        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
        """
        super().__init__()
        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.scale = hidden ** 0.5              

    def forward(self, x):
        """Softmax(xQ @ xK^T) @ xV

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        # YOUR CODE STARTS HERE (can be implemented in ~4 lines)

        query = self.q(x)  # [B,S,C] @ [C,H] => [B,S,H]
        key = self.k(x)    # [B,S,C] @ [C,H] => [B,S,H]
        value = self.v(x)  # [B,S,C] @ [C,H] => [B,S,H]

        scores = query @ key.transpose(-2, -1)  # [B,S,C] @ [B,C,S] => [B,S,S]
        scaled_scores = scores / self.scale  # [B,S,S]
        probs = F.softmax(scaled_scores, dim=-1)  # [B,S,S]
        out = probs @ value  # [B,S,S] @ [B,S,H] => [B,S,H]

        return out
    
        # YOUR CODE ENDS HERE


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0):
        """
        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
            num_heads: int, number of attention heads, should be a divisor of hidden
            causal: use causal masking (do not allow target to look to the future or current token of source)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, x, return_attention=False):
        """Computes [Softmax(x Q_1 @ x K_1^T) @ x V_1 : ... : Softmax(x Q_heads @ x K_heads^T) @ x V_heads] @ U

        or in more details:
        [SelfAttention_1(x) : ... : SelfAttention_h(x)] @ U

        where SelfAttention(x) = Softmax(x Q @ x K^T) @ x V
        and [:] is a concatenation operation.

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        bs, seq, _ = x.shape
 
        # YOUR CODE STARTS HERE
        query = torch.cat([self.q(x) for _ in range(self.num_heads)], dim=0)  # [B,S,C] @ [C,H] => [HEADS,B,S,H]
        key = torch.cat([self.k(x) for _ in range(self.num_heads)], dim=0)    # [B,S,C] @ [C,H] => [HEADS,B,S,H]
        value = torch.cat([self.v(x) for _ in range(self.num_heads)], dim=0)  # [B,S,C] @ [C,H] => [HEADS,B,S,H]
        score = query @ key.transpose(-2, -1)  # [HEADS,B,S,H] @ [HEADS,B,H,S] => [HEADS,B,S,S]
        if self.causal:
            mask = torch.tril(torch.ones(seq, seq)).T
            score = score.masked_fill(mask == 0, float('-inf'))  # [HEADS,B,S,S]
        # ============
        # 2.3.1. compute probabilities naming them probs, first dimension to be NUM_HEADS * B
        scaled_score = (score / self.scale).reshape(self.num_heads * bs, seq, seq)  # [HEADS * B, S, S]
        probs = F.softmax(scaled_score, dim=-1).reshape(self.num_heads * bs, seq, seq)  # [HEADS * B, S, S]
        # ============
        # 2.3.2 apply dropout
        probs = self.dropout(probs)
        # ============
        # 2.3.3 compute attention
        attention = probs @ value  # [HEADS * B, S, S] * [HEADS * B, S, H] => [NUM_HEADS * B, S, H]
        # ============
        # 2.3.4 apply transforms getting to [batch, seq, hidden]
        heads_reshaped = attention.reshape((self.num_heads, bs, seq, -1))  # [NUM_HEADS, B, S, H]
        heads_summed = heads_reshaped.sum(dim=0)  # [B, S, H]
        # ============
        # 2.3.5 mix attentions naming output att
        att = self.mix(heads_summed)  # [B, S, H]

        # YOUR CODE ENDS HERE

        if return_attention:
            return att, probs

        return att
