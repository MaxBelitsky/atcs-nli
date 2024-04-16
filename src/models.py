import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MeanEmbedder(nn.Module):
    def __init__(self, vectors):
        super(MeanEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]

        # Mask all unknown tokens so that 0s don't affect the mean
        mask = embeddings.sum(dim=-1) != 0
        out = embeddings.sum(dim=1) / mask.sum(dim=-1).view(embeddings.shape[0], 1)

        # out = embeddings.mean(dim=1) # [bs, embed_dim]
        return out


class LSTMEmbedder(nn.Module):
    def __init__(self, vectors, n_hidden=2048):
        super(LSTMEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.embedding_dim = self.embedding.embedding_dim
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, batch_first=True)

    def forward(self, x):
        embeddings = self.embedding(x['input_ids']) # [bs, tokens, embed_dim]

        # Pack the embeddings
        packed_embeddings = pack_padded_sequence(
            embeddings,
            x["length"],
            batch_first=True,
            enforce_sorted=False,
        )

        # Pass through LSTM
        packed_output, (hn, cn) = self.lstm(packed_embeddings)

        # Unpack the output
        # output = pad_packed_sequence(packed_output, batch_first=True)[0]

        # return output[:, -1, :] # return the hidden state of the last token
        return hn.squeeze() # return the hidden state of the last token


class BiLSTMEmbedder(nn.Module):
    def __init__(self, vectors):
        super(BiLSTMEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.embedding_dim = self.embedding.embedding_dim
    
    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        # TODO: finish


class BiLSTMPooledEmbedder(nn.Module):
    def __init__(self, vectors):
        super(BiLSTMPooledEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        # TODO: finish


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedder, hidden_dim, n_classes):
        super(SentenceClassificationModel, self).__init__()

        self.embedder = embedder

        n_features = embedder.n_hidden * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, premise, hypothesis):
        premise_embeddings = self.embedder(premise)
        hypothesis_embeddings = self.embedder(hypothesis)

        elem_prod = premise_embeddings * hypothesis_embeddings
        abs_diff = torch.abs(premise_embeddings - hypothesis_embeddings)

        features = torch.cat([premise_embeddings, hypothesis_embeddings, elem_prod, abs_diff], dim=1)

        logits = self.mlp(features)
        return logits
