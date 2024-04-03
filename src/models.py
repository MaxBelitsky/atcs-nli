from torch import nn


class MeanEmbedder(nn.Module):
    def __init__(self, vectors):
        super(MeanEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        out = embeddings.mean(dim=1) # [bs, embed_dim]
        return out


class LSTMEmbedder(nn.Module):
    def __init__(self, vectors):
        super(LSTMEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        # TODO: finish


class BiLSTMEmbedder(nn.Module):
    def __init__(self, vectors):
        super(BiLSTMEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
    
    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        # TODO: finish


class BiLSTMPooledEmbedder(nn.Module):
    def __init__(self, vectors):
        super(BiLSTMPooledEmbedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)

    def forward(self, x):
        embeddings = self.embedding(x) # [bs, tokens, embed_dim]
        # TODO: finish


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedder, hidden_dim, n_classes):
        super(SentenceClassificationModel, self).__init__()

        self.embedder = embedder

        self.mlp = nn.Sequential(
            nn.Linear(embedder.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        embeddings = self.embedder(x)
        logits = self.mlp(embeddings)
        return logits
