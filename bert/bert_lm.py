import torch
from bert.bert_model import BERT
from bert.next_sentence_prediction import NextSentencePrediction
from bert.masked_language_model import MaskedLanguageModel

class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)

    def to(self, device):
        super().to(device)
        self.bert = self.bert.to(device)
        self.next_sentence = self.next_sentence.to(device)
        self.mask_lm = self.mask_lm.to(device)
        return self 