# Summary and Elaboration
We make a from-scratch transformer, built with PyTorch fundamentals and a custom optimization loop. The project is for educational purposes, and is neither fast, efficient nor high fidelity compared to state of the art solutions.

The transformer is trained with unsupervised next-token prediction on the entire CNN and Daily Mail [dataset](https://arxiv.org/abs/1704.04368). This amounts to 287113 articles, around 1.3B characters. This data set features articles with highlights that summarize the body text, making it easy to make a language model (LM) for summerization (body text to highlight,) or elaboration (highlight to body text). This LM is trained to do both functions simultaneously.

This project also features several technical implementations, such as [data cleaning](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/training/data_cleaning.ipynb), a custom [BPE tokenizer](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/training/tokenize_corpus.ipynb) implemented in NumPy, proper handling of stop token causal masking and [interpretability](https://github.com/KristianWold/sum-and-elab/tree/main/notebooks/interpretability) analysis on the level of word embeddings. 


## Training
Training, including data cleaning and tokenization, can be found in [training](https://github.com/KristianWold/sum-and-elab/tree/main/notebooks/training).

We train the transformer on the whole of the predefined training set, for up to five epochs.

## Inference

We test the full model, regularized and trained for five epochs, on an out-of-distribution text fethced from the test corpus. We let the model predict a highlight from the article, and vise versa.

See the [notebook](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/test_data.ipynb) for details about inference, and [here](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/readme.md) for more tests. The articles has been chosen for briefness, but has otherwise not been cherry picked.

## Ebola Infection

### Real Article: 
(cnn)five americans who were monitored for three weeks at an omaha, nebraska, hospital after being exposed to ebola in west africa have been released, a nebraska medicine spokesman said in an email wednesday. one of the five had a heart-related issue on saturday and has been discharged but hasn't left the area, taylor wilson wrote. the others have already gone home. they were exposed to ebola in sierra leone in march, but none developed the deadly virus. they are clinicians for partners in health, a boston-based aid group. they all had contact with a colleague who was diagnosed with the disease and is being treated at the national institutes of health in bethesda, maryland. as of monday, that health care worker is in fair condition. the centers for disease control and prevention in atlanta has said the last of 17 patients who were being monitored are expected to be released by thursday. more than 10,000 people have died in a west african epidemic of ebola that dates to december 2013, according to the world health organization. almost all the deaths have been in guinea, liberia and sierra leone. ebola is spread by direct contact with the bodily fluids of an infected person.

### Real Highlight: 
17 americans were exposed to the ebola virus while in sierra leone in
march. another person was diagnosed with the disease and taken to hospital in
maryland. national institutes of health says the patient is in fair condition
after weeks of treatment.

### Highlight Predicted from Article:

five americans infected with ebola in west africa have been released, a nebraska health spokesman says. one of the five has a heart-related issue. the hospital has said the last of 17 patients who were treated were expected to be released by thursday. one of the five is in fair condition, and has been discharged but hasn't left the area.

**Comments**

The predicted highlight is broadly correct and good language.

### Article Predicted from Highlight:
(cnn) -- four americans who were exposed to the deadly ebola virus in liberia, the health department said monday. one of the infected was in a tent at the liberian capital of san antonio, according to the centers for disease control and prevention. one of the men was treated at a medical center in bellevue, maryland, and is now in a stable condition, the health department said. the second man, who was treated at a medical center in maryland in late march, is in fair condition, the health department said. the other two is in stable condition. the other two are with suspected cases of ebola in the united states. the latest patient, who was treated at a hospital in monrovia, maryland, is in fair condition, the health department said. the two have been identified as 21-year-old sierra leonean thomas eric duncan. duncan is in a critical condition and is in a stable condition, the health department said. he is in a stable condition with three other patients who have died in the west african country, the health department said. the case has been referred to the national institutes of health.

**Comments**

Correct information:
 - Taken to hospital in Maryland
 - Patient in fair condition

Incorrect information:
 - Exposed in Liberia (was Sierra Leone)
 - "Liberian capital of San Antonio", incorrect statement.

Invented information:
 - Four americans exposed
 - Treated in Bellevu in Maryland 
 - Identification of patients
 - patients that died

Other Comments: 
 - Language is good, not repetitive.
 - "the two have been identified as...", but then only one person is identified.

## Word Relations

When predicting highlights and articles on test data, the model did several interesting connections and misconnections between words, as pointed out in the [analysis](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/README.md). The following relations were observed:

- Inferring that Luton is in London
- Conflating Sierra Leone and Liberia
- Inventing that a terrorist suspect was from Syria and named Rashid

Since these inferences can't be localized directly from the test data, we investigate the word embeddings of the model to see if we can explain the interesting inferrences. Specifically, we check cosine similarity between target word embeddings with the embeddings of the whole vocabulary to see what concepts the model might associate. 

The computing the similarity of "Luton" to the vocabulary, "Londons" and "London's" score rank 200 and 297, showing they are significantly more correlated than most of the vocabulary. Similarly, "Luton" score rank 140 with respect to London. Thus, Luton and London have in terms of embedding some overlap of information, making it possible for the model to derive one from the other.

Even more correlated, "Sierra" and "Liberia" score rank 1 and 14 with respect to each other, making them almost equal in terms of embedding. Thus, the model can likely easily conflate them, as observed.

Finally, "Syria" and "terror" (and variations such terrorist and terrorism) score ranks 130, 186 and 189, and 229. In addition, "terror" is strongly correlated middle eastern associations, such as "jihadist" at rank five, and islamist at rank 11. This might help explain why the name "Rashid", a common Syrian name, was predicted by the model. Likely, CNN and Daily Mail report on a lot of conflict and terrorism in the Middle East, providing data to support such correlations in the models. While the data itself is not erronous, the correlations are derived through a specific lens set by the reporting of the newspapers. We observe that such correlations can manifest as extrapolations made by the transformer during inference, resulting in claims that does not generally reflect upon reality. These are typically called transformer hallucinations. 

## Word Clustering

We look for interpretable structures in the word embedding of the transformer at different points during training, with and without regularization. We perform a KMeans clustering on the word vectors directly (using square-sum distance), finding 300 centroids that best cluster the space of word vectors. From these, we select the tightest clusters and retrieve the ten word vectors closest to their respective centroid. We finally check if these groupings of words offer any semantic meaning.

### With Regularization 

#### One epoch:

Cluster 1: 600 900 400 700 300 750 450 350 800 250

Cluster 2: 4pm 10am 5pm 2pm 1pm 7pm 11pm 11am 7am 5am

Cluster 3: 90s 70s 80s eighties 60s 1960s 1950s 1970s 1940s 1980s

Cluster 5: mexico puerto arizona mexican florida nevada texas cuba chicago california

**Comments**
The first three clusters are semantically tight and very simple. They encode different numerical values, the first being "hundreds", second is "time of day", and third is "decades".

The fift is conceptually more interesting, though it is not as semantically tight. They exhibit locations, roughly American south states, but with exceptions. Some are cities, some are countries. 

#### Five epochs:

Cluster 1: protests demonstrations demonstrators protesters protestors protester rallies protest clashes riots

Cluster 2: adelaide melbourne brisbane sydney queensland canberra perth tasman nsw sydney's

Cluster 3: 22 14 13 15 23 26 17 12 25 16

Cluster 10: shouting yelling chanting screaming sobbing waving crying chants cheering singing

**Comments**

After many more epochs of training, the clusters seem to be conceptually tighter and encode more complex concepts. First cluster is a collection of synonyms for "protests". Second is mostly cities in Australia, with the exception of Tasman(ia) and NSW, which are Australian states. Cluster 10 express verbs for making sound in an emotional way.


### Without Regularization 

### One epoch:

1: stopp spi convers consid schweinste mosqu vett contam venez magistr

2: something somet whe odd wh too and nothing dig eight

3: 600 900 700 400 300 450 750 800 250 350

8: july august october february december november september january june april

**Comments**
We observe a big difference in the clusters for the regularized and unregluarized models, espesially when trained for a short amount of time. Without regularization, the first two clusters are very semantically loose, more reminiscient of a "bag of words." Even worse, many of the members are not even full words, but rather subwords.

As regularization decrease the effective dimmension of the embeddings and force more robust encoding, a tighter and more effective representation of the concepts is incentivied. Presumably, this increase the performance of the final model, as it can more easily correlate relevant concepts during inference. 

### Three Epochs:

1: 600 900 400 450 350 700 750 250 300 800

3: weekend's saturday's sunday's sundays thursday's friday's week's saturdays tuesday's wednesday's

5: i ive im my we i'm i've i'd you id

8: insects ants spiders insect bugs squirrel rats bees rept rabbits

**Comments**
With more training, the clusters become more semantically tight, even when unregularized. For example, clusters of subwords are not to be found among top ten clusters. However, the clusters continue to be somewhat semantically diffuse. For example, cluster eight a is rought collection of creatures, pairing both "ants" and "spiders", but also "rats" and "squirrels". The cluster is perhaps reminiscent of a "vermin" cluster. However, "bees" are not typically though of as vermin.




