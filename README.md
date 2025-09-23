# Summary and Elaboration
A from-scratch transformer is built with PyTorch fundamentals and a custom optimization loop. The project is for educational purposes, and is neither fast, efficient nor high fidelity compared to state of the art solutions.

The transformer is trained with unsupervised next-token prediction on the entire CNN and Daily Mail [dataset](https://arxiv.org/abs/1704.04368). This data set features articles with highlights that summarize the body text, 
making it easy to make a language model (LM) for summerization (body text to highlight,) or elaboration (highlight to body text). This LM is trained to do both functions simultaneously.

Further, this project also feautres data cleaning, a custom BPE tokenizer implemented in NumPy, and interpretability analysis on the level of word embeddings. 

## Inference


