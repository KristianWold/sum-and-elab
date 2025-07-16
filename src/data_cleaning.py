import unicodedata
import os
import re

def normalize_to_ascii(text):
    # 1) Decompose Unicode characters
    text_normalized = unicodedata.normalize('NFKD', text)

    # 2) Drop the non-ASCII combining marks
    ascii_bytes = text_normalized.encode('ascii', 'ignore')
    return ascii_bytes.decode('ascii')


def clean_whitespace(text):
    rcw = re.compile(r"\s+") # remove consecutive whitespace
    text = text.replace("\n", " ").replace("\r", " ") # replace newlines with space
    text = rcw.sub(" ", text).strip() # remove leading/trailing whitespace

    return text.lower()


def word_split(line):
    
    normalized_line = normalize_to_ascii(line)
    # Split into words
    word_list = normalized_line.strip().split()
    word_list = [list(word) for word in word_list]
    return word_list

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

        
        

