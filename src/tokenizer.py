import numpy as np
import unicodedata
import re
import random
import torch
from tqdm.notebook import tqdm


class TokenizerChar:
    def __init__(self, corpus):

        # Flatten the list of lists into a single list
        corpus_flatten = []
        for line in corpus:
            words = list(line)
            for word in words:
                corpus_flatten.extend(word)
    
        self.vocab = sorted(list(set(corpus_flatten)))
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.lookup = torch.full((128,), -1, dtype=torch.long)

    
    def encode(self, text):
        return [self.token_to_idx.get(ch, -1) for ch in text]

    def decode(self, indices):
        return "".join([self.vocab[i] for i in indices if i < self.vocab_size])


class TokenizerBPE:
    def __init__(self, corpus, num_merges, ratio=None, verbose=False):

        print("Create character tokenizer")
        self.tokenizer_char = TokenizerChar(corpus)
        
        self.token_to_idx = self.tokenizer_char.token_to_idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.token_freq = {}

        self.vocab_size = self.tokenizer_char.vocab_size

        self.pre_merge_list = []
        self.add_special_tokens(["<>"]) # separator token
        self.sep_token = self.token_to_idx["<>"]

        corpus_flatten = " ".join(corpus)
        del corpus

        print("Split corpus into words")
        corpus_flatten = re.findall(r"\s*[\w']+|[^\w]", corpus_flatten)

        # shuffle and sample
        random.shuffle(corpus_flatten)
        length = len(corpus_flatten)

        if not ratio is None:
            corpus_flatten = corpus_flatten[:int(length * ratio)]

        corpus_flatten = "<>".join(corpus_flatten)
        print("Char tokenize corpus")
        corpus_tokens = self.tokenizer_char.encode(corpus_flatten)
        print("Pre-merge corpus")
        corpus_tokens = self.pre_merge(corpus_tokens)


        print("Merging started")
        self.merge_list = []
        for i in tqdm(range(num_merges)):
            corpus_tokens = self.merge(corpus_tokens, verbose)

        print("Merging complete")


    def encode(self, text, verbose=False, pre_merge=True):

        if verbose:
            decorator = tqdm
        else:
            decorator = lambda x: x

        indices = np.array(self.tokenizer_char.encode(text))
        if len(self.pre_merge_list) > 0 and pre_merge:
            indices = self.pre_merge(indices)

        for (idx1, idx2), new_idx in decorator(self.merge_list):
            slice = np.where(np.logical_and(indices[:-1] == idx1,  indices[1:] == idx2))
            if len(slice[0]) > 0:
                indices[:-1][slice] = new_idx
                indices = np.delete(indices, (slice[0]+1))

        return indices


    def decode(self, indices):
        return "".join([self.idx_to_token[i] for i in indices if i < self.vocab_size])


    def merge(self, corpus_tokens, verbose=False):
        corpus_tokens = np.array(corpus_tokens)  

        new_idx = self.vocab_size
        (idx1, idx2), counts = pair_freq(corpus_tokens, self.sep_token, self.vocab_size)
        self.merge_list.append([(idx1, idx2), self.vocab_size])

        token1 = self.idx_to_token[idx1]
        token2 = self.idx_to_token[idx2]
        if verbose:
            print(f"Merging tokens: {token1} + {token2} -> {new_idx} with count {counts}")
        # Create new token
        new_token = token1 + token2
        self.token_to_idx[new_token] = new_idx
        self.idx_to_token[new_idx] = new_token
        self.token_freq[new_token] = counts
        self.vocab_size += 1

        slice = np.where(np.logical_and(corpus_tokens[:-1] == idx1, corpus_tokens[1:] == idx2))
        if len(slice[0]) > 0:
            corpus_tokens[:-1][slice] = new_idx
            corpus_tokens = np.delete(corpus_tokens, (slice[0]+1))

        return corpus_tokens
    
    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.token_to_idx:
                token_indices = self.tokenizer_char.encode(token)
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.pre_merge_list.append([token_indices, self.vocab_size])
                self.vocab_size += 1
    

    def pre_merge(self, indices):
        """ Pre-merge special token encoding from char tokens"""    

        for token, value in self.pre_merge_list:
            length = len(token)
            token = np.array(token).reshape(-1,1)
            indices_list = []
            for i in range(length):
                indices_list.append(np.pad(indices, (length - i - 1, i), 'constant', constant_values=(0, 0)))
            indices_list = np.array(indices_list)

            locs = np.where(np.equal(indices_list, token).all(axis=0))[0] - length + 1
            del_idx = []
            for loc in list(reversed(locs)):
                indices[loc] = value
                del_idx.extend(list(range(loc + 1, loc + length)))

            if len(del_idx) > 0:    
                del_idx = np.array(del_idx)
                indices = np.delete(indices, del_idx)

        return indices


def pair_freq(indices, sep_token, vocab_size):
    indices = np.array(indices)
    mask = (indices[:-1] == sep_token) + (indices[1:] == sep_token)

    indices_large = indices[:-1] + indices[1:]*vocab_size

    indices_large = indices_large[~mask]
    counts = np.bincount(indices_large)
    temp = np.argmax(counts)
    count = counts[temp]
    idx1 = temp % vocab_size
    idx2 = temp // vocab_size

    return (idx1, idx2), count