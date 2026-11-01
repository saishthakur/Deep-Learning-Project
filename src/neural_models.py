"""Compact PyTorch CNN and LSTM architectures for the shared ISOT task."""
from __future__ import annotations

import re
from collections import Counter

import torch
from torch import nn


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold())


def build_vocabulary(texts, max_size: int = 30_000) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    # 0 is padding and 1 is the unknown token.
    return {token: index + 2 for index, (token, _) in enumerate(counts.most_common(max_size - 2))}


def encode_text(text: str, vocabulary: dict[str, int], max_length: int = 400) -> list[int]:
    ids = [vocabulary.get(token, 1) for token in tokenize(text)[:max_length]]
    return ids + [0] * (max_length - len(ids))


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 128, filters: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(nn.Conv1d(embedding_dim, filters, k) for k in (3, 4, 5))
        self.dropout = nn.Dropout(0.5)
        self.hidden = nn.Linear(filters * 3, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convolutions]
        x = self.dropout(torch.cat(pooled, dim=1))
        return self.output(torch.relu(self.hidden(x))).squeeze(1)


class TextLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 128, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.hidden = nn.Linear(hidden_size, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input_ids):
        lengths = input_ids.ne(0).sum(1).clamp(min=1).cpu()
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        x = self.dropout(hidden[-1])
        return self.output(torch.relu(self.hidden(x))).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

