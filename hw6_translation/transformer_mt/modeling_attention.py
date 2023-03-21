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


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False):
        """Multi-head attention module which computes [softmax(xQ_h @ xK_h^T) @ xV: ...] @ U

        Can work as both self-attention or cross-attention (if kv is provided to .forward).

        Args:
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

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, q, kv=None, key_padding_mask=None, return_attention=False):
        """[Softmax(source Q_1 @ target K_1^T) @ target V_1 : ... ) @ x V_heads] @ U

        Performs self-attention if kv is not specified.
        In this case, kv = q and kv_seq_len = query_seq_len.

        Args:
            q: FloatTensor[batch_size, query_seq_len, input_size]
            kv (target) : optional, FloatTensor[batch_size, kv_seq_len, input_size]
            key_padding_mask: BoolTensor[batch_size, kv_seq_len] 0 means unpadded, 1 means padded

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """

        # Task 1.1 (1 point)
        # Update this function with cross-attention mechanism
        # If target is None, then target (kv) and source (q) will be same.
        # Define k, q, v using self.k, self.q and self.v based on if the target exists or not 
        # Note : Please write shape of each tensor for each line of code
        ## YOUR CODE STARTS HERE## ~ 2 lines code

        if kv is None:
            kv = q
        q = self.q(q)   # BATCH, SRC_SEQ -> BATCH, SRC_SEQ, HIDDEN (where HIDDEN = N_HEADS * HEADSIZE)
        k = self.k(kv)  # BATCH, TAR_SEQ -> BATCH, TAR_SEQ, HIDDEN (where HIDDEN = N_HEADS * HEADSIZE)
        v = self.v(kv)  # BATCH, TAR_SEQ -> BATCH, TAR_SEQ, HIDDEN (where HIDDEN = N_HEADS * HEADSIZE)

        # YOUR CODE ENDS HERE

        bs, attending_seq, _ = q.shape
        attended_seq = k.shape[1]

        # COPY YOUR PREVIOUS HOMEWORK CODE HERE

        k = k.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()  # [batch * num_heads, seq, hidden / num_heads]
        q = q.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()

        # we have [BATCH, SEQ, HIDDEN]
        # we want [BATCH, HEADS, SEQ, HEADSIZE]
        # gather the dimensions (clarifying)
        BATCH, HEADS, SRC_SEQ, TAR_SEQ, HEADSIZE = bs, self.num_heads, attending_seq, attended_seq, self.head_size
        # print(f"BATCH {BATCH} HEADS {HEADS} SRC_SEQ {SRC_SEQ} TARGET_SEQ {TAR_SEQ} HEADSIZE {HEADSIZE}")

        # reshape
        # q = torch.reshape(q, (BATCH, SRC_SEQ, HEADS, HEADSIZE))  # [BATCH, SRC_SEQ, HEADS, HEADSIZE]
        # q = torch.transpose(q, -2, -3)                           # [BATCH, HEADS, SRC_SEQ, HEADSIZE]
        # k = torch.reshape(k, (BATCH, TAR_SEQ, HEADS, HEADSIZE))  # [BATCH, TAR_SEQ, HEADS, HEADSIZE]
        # k = torch.transpose(k, -2, -3)                           # [BATCH, HEADS, TAR_SEQ, HEADSIZE]

        scores = q @ k.transpose(-2, -1)  # [B, H, SRC_SEQ, HEADSIZE] * [B, H, HEADSIZE, TAR_SEQ] -> [B, H, SRC_SEQ, TAR_SEQ]
        # scores = scores.reshape(BATCH * HEADS, SRC_SEQ, TAR_SEQ)  # [BATCH * HEADS, SRC_SEQ, TAR_SEQ]
        # MY CODE ENDS HERE

        assert scores.shape == (bs * self.num_heads, attending_seq, attended_seq)

        if key_padding_mask is not None:
            # Task 1.2 (1 point)
            # Padding
            # Set the scores corresponding to padded positions (key_padding_mask == 1) to -inf
            # 
            # You might need to reshape the scores to [batch_size, seq_len, seq_len]
            # in this case, remember to reshape them back
            # Our implementation is 3 lines
            # YOUR COD E STARTS HERE

            print(f"scores at start should be {BATCH * HEADS} {SRC_SEQ} {TAR_SEQ} is {scores.shape}")
            print(f"key_passing_mask.shape should be {BATCH} {TAR_SEQ} is {key_padding_mask.shape}")
            # print(f"{key_padding_mask}")
            print()
            scores = scores.reshape(BATCH, HEADS, SRC_SEQ, TAR_SEQ)
            print(f"scores after 1... should be {BATCH} {HEADS} {SRC_SEQ} {TAR_SEQ} is {scores.shape}")
            scores = scores.transpose(0, 2)
            print(f"scores after 2... should be {SRC_SEQ} {HEADS} {BATCH} {TAR_SEQ} is {scores.shape}")
            # scores = scores.mul(key_padding_mask)
            # print(f"scores.shape 3a... should be {SRC_SEQ} {HEADS} {BATCH} {TAR_SEQ} is {scores.shape}")
            # print(key_padding_mask==1)
            # scores = torch.Tensor.masked_fill(mask=(key_padding_mask == 1), value=float('-inf'))
            scores = scores.masked_fill(mask=(key_padding_mask == 1), value=float('-inf'))
            # print(scores)
            # assert False, "hold up"
            print(f"scores.shape 3b... should be {SRC_SEQ} {HEADS} {BATCH} {TAR_SEQ} is {scores.shape}")
            scores = scores.transpose(0, 2)
            print(f"scores after 4... should be {BATCH} {HEADS} {SRC_SEQ} {TAR_SEQ} is {scores.shape}")
            scores = scores.reshape(BATCH * HEADS, SRC_SEQ, TAR_SEQ)
            print(f"scores after 5... should be {BATCH * HEADS} {SRC_SEQ} {TAR_SEQ} is {scores.shape}")


            # YOUR CODE ENDS HERE

        assert scores.size() == (bs * self.num_heads, attending_seq, attended_seq),\
            f"scores have wrong shape. Expected {(bs * self.num_heads, attending_seq, attended_seq)}, got {scores.size()}"

        if self.causal:
            causal_mask = torch.triu(torch.ones(attending_seq, attended_seq, dtype=torch.bool, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask.bool().unsqueeze(0), float("-inf"))

        # COPY YOUR PREVIOUS HOMEWORK CODE HERE (~ 4 lines)

        scores = scores / self.scale  # [BATCH * HEADS, SRC_SEQ, TAR_SEQ]
        probs = nn.functional.softmax(scores, dim=-1)  # [BATCH * HEADS, SRC_SEQ, TAR_SEQ]
        # probs = self.dropout(probs)
        att = probs @ v  # [BATCH * HEADS, SRC_SEQ, TAR_SEQ] @ [BATCH, TAR_SEQ, HIDDEN] => [BATCH, HEADS, SRC_SEQ, HIDDEN]
        att = att.transpose(1, 2)  # [BATCH, SRC_SEQ, HEADS, HIDDEN]
        att = att.reshape(BATCH, -1, SRC_SEQ)  # [BATCH, SRC_SEQ * HEADS, HIDDEN]
        att = att.transpose(1, 2)  # [BATCH, HIDDEN, SRC_SEQ * HEADS]
        att = att.contiguous()

        att = self.mix(att)  # [BATCH, HIDDEN, SRC_SEQ * HEADS]
        # MY CODE ENDS HERE

        if return_attention:
            return att, probs

        return att
