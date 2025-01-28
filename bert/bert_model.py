import torch
from bert.embedding import BERTEmbedding
from bert.encoder_layer import EncoderLayer

class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1, seq_len=512):
        """
        :param vocab_size: vocab_size of total words
        :param d_model: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param heads: number of attention heads
        :param dropout: dropout rate
        :param seq_len: maximum sequence length
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=seq_len)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

    def to(self, device):
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.encoder_blocks = torch.nn.ModuleList([layer.to(device) for layer in self.encoder_blocks])
        return self 