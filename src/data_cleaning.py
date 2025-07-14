import unicodedata
import os
import numpy as np
import re

import textwrap
import ipywidgets as widgets
from IPython.display import display

def normalize_to_ascii(text):
    # 1) Decompose Unicode characters
    text_normalized = unicodedata.normalize('NFKD', text)

    # 2) Drop the non-ASCII combining marks
    ascii_bytes = text_normalized.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')


def clean_text(text):
    rcw = re.compile(r"\s+")
    text = text.replace("\n", " ").replace("\r", " ")
    text = rcw.sub(" ", text).strip()

    return text.lower()


def read_first_n(directory_path, n):
    # List all entries in the directory
    filenames = os.listdir(directory_path)
    # Filter to only .txt files
    txt_files = [f for f in filenames if f.lower().endswith('.story')]
    # Sort alphabetically (or by any other criteria you like)
    #txt_files.sort()
    # Take the first n
    first_n = txt_files[:n]
    
    contents = []
    for fname in first_n:
        full_path = os.path.join(directory_path, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            contents.append(normalize_to_ascii(f.read()))
    return contents


def word_split(line):
    
    normalized_line = normalize_to_ascii(line)
    # Split into words
    word_list = normalized_line.strip().split()
    word_list = [list(word) for word in word_list]
    return word_list


def fused_qa(question_list, answer_list, tokenizer):
    q = tf.convert_to_tensor([[tokenizer.token_to_idx["<q>"]]])
    a = tf.convert_to_tensor([[tokenizer.token_to_idx["<a>"]]])
    sos = tf.convert_to_tensor([[tokenizer.token_to_idx["<s>"]]])
    eos = tf.convert_to_tensor([[tokenizer.token_to_idx["</s>"]]])

    corpus_list = []
    for question, answer in tqdm(list(zip(question_list, answer_list))):
        q_tokens = tokenizer.encode(question)
        a_tokens = tokenizer.encode(answer)
        qa = tf.concat([sos, q, q_tokens, a, a_tokens, eos], axis=1)
        corpus_list.append(qa)
    
    return corpus_list


def split_on_value(tensor, delim):
    # 1) Find delimiter indices
    is_delim   = tf.equal(tensor, delim)
    delim_idxs = tf.where(is_delim)[:, 0]      # shape [num_delims]
    
    # 2) Build start/end indices so that each segment starts at either 0 or a delimiter
    n = tf.shape(tensor)[0]
    starts = tf.concat([[0],         delim_idxs], axis=0)  # e.g. [0, 3, 6]
    ends   = tf.concat([delim_idxs, [n]],       axis=0)  # e.g. [3, 6, 9]
    
    # 3) Compute each segment’s length (end − start)
    lengths = ends - starts                         # e.g. [3, 3, 3]
    
    # 4) Split the original tensor (delimiters are still in it)
    parts = tf.split(tensor, lengths)
    
    # 5) (optional) drop any empty segments (can happen if tensor starts/ends with delim)
    parts = [p for p in parts if tf.shape(p)[0] > 0]
    
    return parts

        
        

