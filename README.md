# Summary and Elaboration
A from-scratch transformer is built with PyTorch fundamentals and a custom optimization loop. The project is for educational purposes, and is neither fast, efficient nor high fidelity compared to state of the art solutions.

The transformer is trained with unsupervised next-token prediction on the entire CNN and Daily Mail [dataset](https://arxiv.org/abs/1704.04368). This data set features articles with highlights that summarize the body text, 
making it easy to make a language model (LM) for summerization (body text to highlight,) or elaboration (highlight to body text). This LM is trained to do both functions simultaneously.

Further, this project also feautres data cleaning, a custom BPE tokenizer implemented in NumPy, and interpretability analysis on the level of word embeddings. 

## Training
Training, including data cleaning and tokenization, can be found in [training](https://github.com/KristianWold/sum-and-elab/tree/main/notebooks/training).

We train the transformer on the whole of the predefined training set, for up to five epochs.

## Inference

Highlight: london's metropolitan police say the man was arrested at luton airport after landing on a flight from istanbul. he's been charged with terror offenses allegedly committed since the start of november.

Predicted Article: (cnn) -- a man who was arrested at luton airport on wednesday after landing on a flight from istanbul said he was arrested and questioned by an immigration officer. the man, who did not want to be named, was detained by immigration authorities. the man is charged with terrorism offenses, according to immigration and customs enforcement (fpf). he was booked on an unrelated charge. the man, originally from syria, was arrested wednesday at luton airport, where he boarded a flight from istanbul. he was charged with terror offenses, and the flight landed at luton international airport. he was arrested with suspected terrorism offenses, according to the airport. the latest charge is connected to the alleged plot in istanbul, the airport's website says.

Actual Article: london (cnn)a 19-year-old man was charged wednesday with terror offenses after he was arrested as he returned to britain from turkey, london's metropolitan police said. yahya rashid, a uk national from northwest london, was detained at luton airport on tuesday after he arrived on a flight from istanbul, police said. he's been charged with engaging in conduct in preparation of acts of terrorism, and with engaging in conduct with the intention of assisting others to commit acts of terrorism. both charges relate to the period between november 1 and march 31. rashid is due to appear in westminster magistrates' court on wednesday, police said. cnn's lindsay isaac contributed to this report.

Recovered Highlight: a man on a flight from istanbul says he was arrested and questioned by immigration officers. he boarded the flight from istanbul to luton, authorities say. the man is charged with terrorism offenses, according to the airport's website.

## Interpretability

### With Regularization 

One epoch:

1: 600 900 400 700 300 750 450 350 800 250

2: 4pm 10am 5pm 2pm 1pm 7pm 11pm 11am 7am 5am

3: 90s 70s 80s eighties 60s 1960s 1950s 1970s 1940s 1980s

5: mexico puerto arizona mexican florida nevada texas cuba chicago california

Five epochs:

1: protests demonstrations demonstrators protesters protestors protester rallies protest clashes riots

2: adelaide melbourne brisbane sydney queensland canberra perth tasman nsw sydney's

3: 22 14 13 15 23 26 17 12 25 16

10: shouting yelling chanting screaming sobbing waving crying chants cheering singing

### Without Regularization 

One epoch:

1: stopp spi convers consid schweinste mosqu vett contam venez magistr

2: something somet whe odd wh too and nothing dig eight

3: 600 900 700 400 300 450 750 800 250 350

8: july august october february december november september january june april

Three Epochs:

1: 600 900 400 450 350 700 750 250 300 800

3: weekend's saturday's sunday's sundays thursday's friday's week's saturdays tuesday's wednesday's

5: i ive im my we i'm i've i'd you id

8: insects ants spiders insect bugs squirrel rats bees rept rabbits




