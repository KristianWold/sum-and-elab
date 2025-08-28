# Summary and Elaboration
This repo features a from-scratch transformer built with pytorch fundamentals and custom optimization. The project is for educational purposes, and is neither fast, efficient nor high fidelity compared to state of the art solutions.

The transformer is trained with unsupervised next-token prediction on the entire CNN and Daily Mail [dataset](https://arxiv.org/abs/1704.04368). This data set features articles with highlights that summarize the body text, 
making it easy to make a language model (LM) for summerization (body text to highlight,) or elaboration (highlight to body text). In this project, the LM is trained to do both functions.

In addition to making the transformer, this project feautres data cleaning, a custom BPE tokenizer implemented in NumPy, and interpretability analysis on the level of word embeddings.
