import json
import re
import os
from typing import List, Dict, Tuple

class TurkishTokenizer:
    def __init__(self):
        # Get the directory of the current file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load JSON files into memory
        self.roots = self.load_json("kokler_v05.json")
        self.suffixes = self.load_json("ekler_v05.json")
        self.bpe_tokens = self.load_json("bpe_v05.json")
        
        # Special tokens
        self.special_tokens = {
            "<uppercase>": 0,
            "<space>": 1,
            "<newline>": 2,
            "<tab>": 3,
            "<unknown>": 4,
            "[CLS]": 5,
            "[SEP]": 6,
            "[MASK]": 7,
            "[PAD]": 8
        }

        # Special token attributes for BERT
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"
        self.pad_token_id = 8
        
    def load_json(self, file_path: str) -> Dict[str, int]:
        full_path = os.path.join(self.current_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def tokenize(self, text: str) -> List[str]:
        result = tokenize(text)  # Using the existing tokenize function
        return result["tokens"]
    
    def encode(self, text_or_tokens) -> List[int]:
        if isinstance(text_or_tokens, str):
            result = tokenize(text_or_tokens)
            return result["ids"]
        elif isinstance(text_or_tokens, list):
            # Convert tokens to ids
            ids = []
            for token in text_or_tokens:
                if token in self.special_tokens:
                    ids.append(self.special_tokens[token])
                elif token in self.bpe_tokens:
                    ids.append(self.bpe_tokens[token])
                elif token in self.roots:
                    ids.append(self.roots[token])
                elif token in self.suffixes:
                    ids.append(self.suffixes[token])
                else:
                    ids.append(self.special_tokens["<unknown>"])
            return ids
        
    def decode(self, ids: List[int]) -> str:
        tokens = []
        reverse_vocab = {}
        for k, v in self.bpe_tokens.items():
            reverse_vocab[v] = k
        for k, v in self.roots.items():
            reverse_vocab[v] = k
        for k, v in self.suffixes.items():
            reverse_vocab[v] = k
        for k, v in self.special_tokens.items():
            reverse_vocab[v] = k

        with open("reverse_vocab.json", "w", encoding="utf-8") as f:
            json.dump(reverse_vocab, f, ensure_ascii=False, indent=4)

        make_next_uppercase = False
        
        for id in ids:
            if id in reverse_vocab:
                if id == 0:  # <uppercase> token
                    make_next_uppercase = True
                    continue
                elif id == 1:
                    tokens.append(" ")
                elif id == 2:
                    tokens.append("\n")
                elif id == 3:
                    tokens.append("\t")
                else:
                    token = reverse_vocab[id]
                    if make_next_uppercase:
                        token = token.capitalize()
                        make_next_uppercase = False
                    tokens.append(token)
            else:
                tokens.append("<unknown>")
        
        return "".join(tokens)  # Changed from " ".join to handle spaces properly
    
    def get_vocab(self) -> Dict[str, int]:
        return {**self.roots, **self.suffixes, **self.bpe_tokens, **self.special_tokens}

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load JSON files into memory
def load_json(file_path: str) -> Dict[str, int]:
    full_path = os.path.join(CURRENT_DIR, file_path)
    with open(full_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load roots, suffixes, and BPE tokens
roots = load_json("kokler_v05.json")
suffixes = load_json("ekler_v05.json")
bpe_tokens = load_json("bpe_v05.json")

# Special token IDs
SPECIAL_TOKENS = {
    "<uppercase>": 0,
    "<space>": 1,
    "<newline>": 2,
    "<tab>": 3,
    "<unknown>": 4
}

# Tokenize the input text
def tokenize(text: str) -> Dict[str, List]:
    tokens = []
    ids = []

    # Process each character for special whitespace tokens
    i = 0
    while i < len(text):
        char = text[i]
        if char == ' ':
            tokens.append("<space>")
            ids.append(SPECIAL_TOKENS["<space>"])
        elif char == '\n':
            tokens.append("<newline>")
            ids.append(SPECIAL_TOKENS["<newline>"])
        elif char == '\t':
            tokens.append("<tab>")
            ids.append(SPECIAL_TOKENS["<tab>"])
        elif char.isalnum() or char in '.,!?;':
            # Collect the word or punctuation
            word_start = i
            while i < len(text) and (text[i].isalnum() or text[i] in '.,!?;'):
                i += 1
            word = text[word_start:i]
            i -= 1  # Adjust for the outer loop increment
            
            if any(c.isupper() for c in word):
                # Split by uppercase letters and process each part
                parts = re.split(r'([A-Z][^A-Z]*)', word)
                for part in parts:
                    if part:
                        if part[0].isupper():
                            tokens.append("<uppercase>")
                            ids.append(SPECIAL_TOKENS["<uppercase>"])
                            process_word(part.lower(), tokens, ids)
                        else:
                            process_word(part, tokens, ids)
            else:
                process_word(word, tokens, ids)
        i += 1

    return {"tokens": tokens, "ids": ids}

# Process a single word
def process_word(word: str, tokens: List[str], ids: List[int]):
    # Check if the word matches a root
    root, root_id, remainder = match_root(word)

    if root:
        tokens.append(root)
        ids.append(root_id)
        if remainder:
            process_remainder(remainder, tokens, ids)
    else:
        # Try BPE tokenization
        bpe_success = process_bpe(word, tokens, ids)
        if not bpe_success:
            # If no matches found, mark as unknown
            tokens.append("<unknown>")
            ids.append(SPECIAL_TOKENS["<unknown>"])

# Match a root from the roots dictionary
def match_root(word: str) -> Tuple[str, int, str]:
    for i in range(len(word), 1, -1):
        if word[:i] in roots:
            return word[:i], roots[word[:i]], word[i:]
    return None, None, word

# Process the remainder of the word
def process_remainder(remainder: str, tokens: List[str], ids: List[int]):
    # Check if the remainder matches a suffix
    suffix, suffix_id = match_suffix(remainder)

    if suffix:
        tokens.append(suffix)
        ids.append(suffix_id)
        remainder = remainder[len(suffix):]

        if remainder:
            process_remainder(remainder, tokens, ids)
    else:
        # Check if the remainder matches another root
        root, root_id, remainder = match_root(remainder)
        if root:
            tokens.append(root)
            ids.append(root_id)
            if remainder:
                process_remainder(remainder, tokens, ids)
        else:
            # Try BPE tokenization
            bpe_success = process_bpe(remainder, tokens, ids)
            if not bpe_success:
                # If no matches found, mark as unknown
                tokens.append("<unknown>")
                ids.append(SPECIAL_TOKENS["<unknown>"])

# Match a suffix from the suffixes dictionary
def match_suffix(word: str) -> Tuple[str, int]:
    for i in range(len(word), 0, -1):
        if word[:i] in suffixes:
            return word[:i], suffixes[word[:i]]
    return None, None

# Process a word using BPE
def process_bpe(word: str, tokens: List[str], ids: List[int]) -> bool:
    i = 0
    found_any = False
    while i < len(word):
        found_match = False
        for j in range(len(word), i, -1):
            if word[i:j] in bpe_tokens:
                tokens.append(word[i:j])
                ids.append(bpe_tokens[word[i:j]])
                i = j
                found_match = True
                found_any = True
                break
        if not found_match:
            i += 1
    return found_any

# Example execution
if __name__ == "__main__":
    input_texts = [
        "Kitabı ve defterleri getirn,\nYouTube\t",
        "Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz"
    ]
    for input_text in input_texts:
        print(input_text)
        result = tokenize(input_text)
        print(result)

    """ Example outputs:
    Kitabı ve defterleri getirn,
    YouTube	
    {'tokens': ['<uppercase>', 'kitab', 'ı', '<space>', 've', '<space>', 'defter', 'ler', 'i', '<space>', 'getir', 'n', ',', '<newline>', '<uppercase>', 'you', '<uppercase>', 'tube', '<tab>'], 'ids': [0, 385, 22270, 1, 19901, 1, 2001, 22268, 22269, 1, 159, 22284, 20022, 2, 0, 643, 0, 21941, 3]}
    
    Bir maddenin yanması ile çıkan ve içinde katı zerrelerle buğu bulunan değişik renklerde gaz
    {'tokens': ['<uppercase>', 'bir', '<space>', 'madde', 'nin', '<space>', 'yan', 'ma', 'sı', '<space>', 'ile', '<space>', 'çık', 'a', 'n', '<space>', 've', '<space>', 'için', 'de', '<space>', 'katı', '<space>', 'zerre', 'ler', 'le', '<space>', 'buğu', '<space>', 'bulun', 'a', 'n', '<space>', 'değişik', '<space>', 'renk', 'ler', 'de', '<space>', 'gaz'], 'ids': [0, 1, 1, 175, 22280, 1, 59, 22288, 22286, 1, 19888, 1, 422, 22274, 22284, 1, 19901, 1, 19886, 22277, 1, 926, 1, 5976, 22268, 22281, 1, 13592, 1, 13, 22274, 22284, 1, 273, 1, 564, 22268, 22277, 1, 965]} 
    """
