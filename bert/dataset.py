import torch
from torch.utils.data import Dataset
import random

def collate_batch(batch, device=None):
    """
    Custom collate function for DataLoader that ensures all tensors are on the same device
    """
    batch_size = len(batch)
    
    # Get the first item to determine device if not specified
    if device is None:
        device = batch[0]['bert_input'].device
        
    # Initialize tensors
    max_len = batch[0]['bert_input'].size(0)
    bert_input = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    segment_label = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    bert_label = torch.ones((batch_size, max_len), dtype=torch.long, device=device) * -100
    is_next = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i, item in enumerate(batch):
        bert_input[i] = item['bert_input']
        segment_label[i] = item['segment_label']
        bert_label[i] = item['bert_label']
        is_next[i] = item['is_next']
    
    return {
        'bert_input': bert_input,
        'segment_label': segment_label,
        'bert_label': bert_label,
        'is_next': is_next
    }

class BERTDataset(Dataset):
    """
    Dataset for BERT pre-training
    Prepares masked language model and next sentence prediction tasks
    """
    def __init__(self, corpus_path, tokenizer, seq_len=512, encoding="utf-8", corpus_lines=None, device=None):
        """
        Initialize dataset
        
        Args:
            corpus_path: Path to your text corpus (CSV file)
            tokenizer: Turkish tokenizer instance
            seq_len: Maximum sequence length
            encoding: Text encoding of corpus
            corpus_lines: Number of lines to load (None for all)
            device: Device to put tensors on ('cuda', 'mps', or 'cpu')
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines
        self.encoding = encoding
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Special tokens
        self.cls_token = self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else '[CLS]'
        self.sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else '[SEP]'
        self.mask_token = self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else '[MASK]'
        self.pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0

        # Load corpus
        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = [line.strip() for line in f.readlines()[:corpus_lines]]
            self.corpus_size = len(self.lines)

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, item):
        # Get one sample from corpus consisting of two sentences
        t1, t2, is_next_label = self.get_sentence_pair(item)

        # Convert to ids
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Combine into one sequence
        tokens = [self.cls_token] + t1_random + [self.sep_token] + t2_random + [self.sep_token]
        segment_ids = [0] * (len(t1_random) + 2) + [1] * (len(t2_random) + 1)

        # Convert tokens to ids
        input_ids = self.tokenizer.encode(tokens) if hasattr(self.tokenizer, 'encode') else self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create masked language model labels
        masked_labels = [-100] + t1_label + [-100] + t2_label + [-100]

        # Padding
        padding = [self.pad_token_id] * (self.seq_len - len(input_ids))
        input_ids += padding
        segment_ids += padding
        masked_labels += padding

        # Convert to tensors and move to device
        input_ids = torch.tensor(input_ids[:self.seq_len], device=self.device)
        segment_ids = torch.tensor(segment_ids[:self.seq_len], device=self.device)
        masked_labels = torch.tensor(masked_labels[:self.seq_len], device=self.device)
        is_next = torch.tensor(is_next_label, device=self.device)

        return {
            "bert_input": input_ids,
            "segment_label": segment_ids,
            "bert_label": masked_labels,
            "is_next": is_next
        }

    def get_sentence_pair(self, item):
        """Get next sentence random either from the same doc or from another doc"""
        t1 = self.get_tokens(self.lines[item])

        # 50% chance for IsNext
        if random.random() > 0.5:
            t2 = self.get_tokens(self.lines[random.randint(0, len(self.lines) - 1)])
            is_next_label = 0
        else:
            t2 = self.get_tokens(self.lines[min(item + 1, len(self.lines) - 1)])
            is_next_label = 1

        return t1, t2, is_next_label

    def random_word(self, tokens):
        """
        Masking some random tokens for masked language modeling task
        """
        output_label = []
        output_tokens = []

        vocab_list = list(self.tokenizer.get_vocab().keys()) if hasattr(self.tokenizer, 'get_vocab') else list(self.tokenizer.vocab.keys())

        for token in tokens:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% chance to mask token
                if prob < 0.8:
                    output_tokens.append(self.mask_token)
                # 10% chance to change to random token
                elif prob < 0.9:
                    output_tokens.append(random.choice(vocab_list))
                # 10% chance to keep current token
                else:
                    output_tokens.append(token)

                token_id = self.tokenizer.encode([token])[0] if hasattr(self.tokenizer, 'encode') else self.tokenizer.convert_tokens_to_ids([token])[0]
                output_label.append(token_id)
            else:
                output_tokens.append(token)
                output_label.append(-100)  # -100 index = ignore

        return output_tokens, output_label

    def get_tokens(self, sentence):
        """Tokenize a sentence"""
        tokens = self.tokenizer.tokenize(sentence) if hasattr(self.tokenizer, 'tokenize') else self.tokenizer.encode(sentence)
        return tokens[:self.seq_len - 2]  # -2 for [CLS] and [SEP] 