# Summary and Elaboration
We make a [from-scratch transformer](https://github.com/KristianWold/sum-and-elab/tree/main/src/transformer_kristianwold), built with PyTorch fundamentals and a custom optimization loop. This project is for educational purposes, and is neither fast, efficient nor high fidelity compared to state of the art solutions.

The transformer is trained with unsupervised next-token prediction on the entire CNN and Daily Mail [dataset](https://arxiv.org/abs/1704.04368). This amounts to 287113 articles, around 1.3B characters. This data set features articles with highlights that summarize the body text, making it simple to train a language model (LM) for doing summerization (body text to highlight,) or elaboration (highlight to body text). This LM is trained to do both functions simultaneously.

This project also features several technical implementations, such as [data cleaning](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/training/data_cleaning.ipynb), a custom [BPE tokenizer](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/training/tokenize_corpus.ipynb) implemented in NumPy, proper handling of stop token causal masking and [interpretability](https://github.com/KristianWold/sum-and-elab/tree/main/notebooks/interpretability) analysis on the level of word embeddings. 


## Training
Notebooks for training, including data cleaning and tokenization, can be found [here](https://github.com/KristianWold/sum-and-elab/tree/main/notebooks/training).

The architecture features an embed size of 1152, with 18 transformer block layers, each with 18 attention heads. This totals around 316M parameters. To decrease vocabulary size, we make use of lower-case text only and remove standardize all non-standard ASCII characters. Generally, a fairly limited amount semantic meaning is lost in this process for most text, while slicing roughtly half of the needed unique tokens. The corpus is then tokenized with BPE, creating rougly 24k unique tokens. We train the transformer on the whole of the predefined training corpus for up to five epochs. This took approximatly 35 hours on a RTX4080 16GB ram GPU.

## Inference

We test the model on out-of-distribution text fetched from the test corpus. The model predicts highlights from articles, and vise versa.

See the [notebook](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/test_data.ipynb) for details about inference, and [here](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/README.md) for more results and discussion. The articles tested were chosen for briefness, but has otherwise not been cherry picked. 

## Ebola Infection Story
The following is a CNN news article about the spreading of the Ebola virus.
### Real Article: 
(cnn)five americans who were monitored for three weeks at an omaha, nebraska, hospital after being exposed to ebola in west africa have been released, a nebraska medicine spokesman said in an email wednesday. one of the five had a heart-related issue on saturday and has been discharged but hasn't left the area, taylor wilson wrote. the others have already gone home. they were exposed to ebola in sierra leone in march, but none developed the deadly virus. they are clinicians for partners in health, a boston-based aid group. they all had contact with a colleague who was diagnosed with the disease and is being treated at the national institutes of health in bethesda, maryland. as of monday, that health care worker is in fair condition. the centers for disease control and prevention in atlanta has said the last of 17 patients who were being monitored are expected to be released by thursday. more than 10,000 people have died in a west african epidemic of ebola that dates to december 2013, according to the world health organization. almost all the deaths have been in guinea, liberia and sierra leone. ebola is spread by direct contact with the bodily fluids of an infected person.

### Real Highlight: 
17 americans were exposed to the ebola virus while in sierra leone in march. another person was diagnosed with the disease and taken to hospital in maryland. national institutes of health says the patient is in fair condition after weeks of treatment.

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
 - "Liberian capital of San Antonio"

Invented information:
 - Four americans exposed
 - Treated in Bellevu in Maryland 
 - Identification of patients
 - patients that died

Other Comments: 
 - Language is good, not repetitive.
 - "the two have been identified as...", but then only one person is identified.

## Interpreting Word Embeddings

When predicting highlights and articles on test data, the model did several interesting connections and misconnections between different concepts, as pointed out in the [analysis](https://github.com/KristianWold/sum-and-elab/blob/main/notebooks/inference/README.md). The following relations were observed:

- Correctly inferring that Luton is in London
- Conflating Sierra Leone with Liberia
- Inventing that an actual terrorist suspect was from Syria.

Since these inferences can't be localized directly from the test data, we investigate the word embeddings of the model to see if we can explain the interesting inferences. Specifically, we check Cosine Similarity to measure closeness in encoding between target word embeddings with the embeddings of the whole model vocabulary. Since identical encodings yield identical model inferences, similar encodings likely invoke similar behavior. 

### Correctly inferring that Luton is in London

We compute and rank the Cosine Similariy score of " luton" to the vocabulary. " londons" and " london's" score rank 201 and 298 out of 24k, showing they are significantly more correlated than most of the vocabulary. Conversely, " luton" score rank 140 with respect to " london". Thus, Luton and London have in terms of embedding substantial overlap of information, showing how the model might derived one from the other.

### Conflating Sierra Leone with Liberia

Even closer, " sierra" and " liberia" score rank two and 15 with respect to each other, making them very close in terms of embeddings. Thus, the model can likely easily conflate them, as observed. Why are the embeddings not more distinct in practice, being separate countries? Relations are learned through the data, and we can hypothesise that the training corpus don't substantiate the distinction enough to produce more different embeddings. Checking the training corpus, we find that Sierra Leone and Liberia were mentioned only 3300 and 4600 times, respectivly, compared to 80 500 mentiones for England and 86 000 for USA.

### Inventing that an actual terrorist suspect was from Syria.

Finally, " syria" and " terror" (and variations such terrorist and terrorism) score ranks 130, 186 and 189, and 229. In addition, "terror" score high rank with many typically middle eastern associated concepts, such as "jihadist" at rank five, and "islamist" at rank 11. Likely, CNN and Daily Mail report on a lot of conflict and terrorism in and relating to the Middle East, providing data to support such correlations in the model. [Inspectig the corpus], we find that Syria and "terror" are mentioned 9900 and 21800 times, with a high co-occurance of 4000. This likely leads to similar learned embeddings, connecting the concepts in the behavior of the model. While the data in itself is not erronous, the correlations are derived through a specific lens set by the reporting of these newspapers. From the predicted article, we observe that this can manifest as extrapolations in the form of claims that does not generally reflect upon reality, such as the terrorist hailing from Syria. These are typically called transformer hallucinations. 

## Word Clustering

We look for interpretable structures in the word embedding of the transformer at different points during training, with and without regularization. We perform a KMeans clustering on the word vectors directly (using square-sum distance), finding 300 centroids that best cluster the space of word vectors. From these, we select the tightest clusters and retrieve the ten word vectors closest to their respective centroid. We finally check if these groupings of words offer any semantic meaning.

### With Regularization 

#### One epoch:

| **Cluster** | **Members**                                                                         |
| ----------- | ----------------------------------------------------------------------------------- |
| 1           | 600, 900, 400, 700, 300, 750, 450, 350, 800, 250                                    |
| 2           | 4pm, 10am, 5pm, 2pm, 1pm, 7pm, 11pm, 11am, 7am, 5am                                 |
| 3           | 90s, 70s, 80s, eighties, 60s, 1960s, 1950s, 1970s, 1940s, 1980s                     |
| 5           | mexico, puerto, arizona, mexican, florida, nevada, texas, cuba, chicago, california |


#### Comments

* Clusters 1–3 are semantically tight and simple:
 * Cluster 1: "hundreds"
 * Cluster 2: "time of day"
 * Cluster 3: "decades"
* Cluster 5 is conceptually interesting but looser: locations, mostly U.S. South + exceptions (mix of cities, states, countries).

#### Five epochs:

| **Cluster** | **Members**                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| 1           | protests, demonstrations, demonstrators, protesters, protestors, protester, rallies, protest, clashes, riots |
| 2           | adelaide, melbourne, brisbane, sydney, queensland, canberra, perth, tasman, nsw, sydney’s                    |
| 3           | 22, 14, 13, 15, 23, 26, 17, 12, 25, 16                                                                       |
| 10          | shouting, yelling, chanting, screaming, sobbing, waving, crying, chants, cheering, singing                   |

#### Comments

* Clusters are conceptually tighter and encode richer concepts:

 * Cluster 1: synonyms for protests.

 * Cluster 2: mostly Australian cities, plus states (Tasmania, NSW).

 * Cluster 10: expressive verbs for emotional sounds.

### Without Regularization 

#### One epoch:

| **Cluster** | **Members**                                                                          |
| ----------- | ------------------------------------------------------------------------------------ |
| 1           | stopp, spi, convers, consid, schweinste, mosqu, vett, contam, venez, magistr         |
| 2           | something, somet, whe, odd, wh, too, and, nothing, dig, eight                        |
| 3           | 600, 900, 700, 400, 300, 450, 750, 800, 250, 350                                     |
| 8           | july, august, october, february, december, november, september, january, june, april |

#### Comments
* Clear differences between regularized vs. unregularized clusters, especially after only one epoch.
* Clusters 1 & 2 are semantically loose → more like “bag of words,” with many subwords instead of whole words.
* Without regularization, embeddings spread across fragmented or noisy concepts.
* Regularization reduces effective embedding dimensionality, encouraging tighter, more robust encodings, likely improving model performance by making relevant concepts easier to correlate at inference.

### Three Epochs:

| **Cluster** | **Members**                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| 1           | 600, 900, 400, 450, 350, 700, 750, 250, 300, 800                                                          |
| 3           | weekend's, saturday's, sunday's, sundays, thursday's, friday's, week's, saturdays, tuesday's, wednesday's |
| 5           | i, ive, im, my, we, i'm, i've, i'd, you, id                                                               |
| 8           | insects, ants, spiders, insect, bugs, squirrel, rats, bees, rept, rabbits                                 |

**Comments**
* After three epochs, clusters are semantically tighter, even unregularized.
* Subword noise is largely gone.
* Clusters still somewhat diffuse.
* Cluster 8 = “creatures” / “vermin-like,” mixing insects (ants, spiders) with mammals (rats, squirrels) and outliers like bees (not vermin).
* Suggests progress toward coherence, but without regularization, the boundaries between concepts remain more fuzzy.



