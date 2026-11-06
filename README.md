# SC5002 Lab Assignment 2

**Course:** SC5002 Artificial Intelligence Fundamentals and Applications, Nanyang Technological University

**Author:** Yeo Wan Ling

This repository contains my work for Lab Assignment 2. It has three parts:

- **Exercise 1, K-Means clustering:** grouping 20 customers by income and spending score, choosing the number of clusters with the elbow method, and testing how much the random starting point matters
- **Exercise 2, a K-Means failure case:** why K-Means cannot separate two rings, one inside the other
- **Exercise 3, a neural network:** using the network from Tutorial 7 to tell apart the vowels in *head* and *had* from real speech recordings

Generative AI was used throughout as a learning partner, as the assignment requires. How I used it is described in [Use of Generative AI](#use-of-generative-ai).

## Contents

- [Repository structure](#repository-structure)
- [How to run the notebook](#how-to-run-the-notebook)
- [Exercise 1: K-Means clustering](#exercise-1-k-means-clustering)
- [Exercise 2: Why K-Means fails on rings](#exercise-2-why-k-means-fails-on-rings)
- [Exercise 3: Neural network vowel classification](#exercise-3-neural-network-vowel-classification)
- [Debugging notes](#debugging-notes)
- [Use of Generative AI](#use-of-generative-ai)
- [Key takeaways](#key-takeaways)
- [Submission links](#submission-links)

## Repository structure

```
.
├── README.md                    # This file
├── SC5002_Lab2.ipynb            # Completed Google Colab notebook (all three exercises)
└── figs/                        # Figures saved by the notebook
```

The full written analysis is in the PDF report submitted on NTULearn.

## How to run the notebook

### Option A: Google Colab (recommended)

1. Open the notebook in Colab: **[Open in Colab]([colab-link])**
2. Click **Runtime → Run all**.
3. No setup is needed. The first cell installs `pyreadr` and downloads the vowel dataset automatically. PyTorch and every other library are already in Colab.

Exercise 3 trains a few small networks, so a full run takes about a minute.

### Option B: Run locally

You need Python 3.9 or later.

```bash
pip install numpy pandas matplotlib scikit-learn torch pyreadr jupyter
jupyter notebook SC5002_Lab2.ipynb
```

The first cell uses Colab's `!wget` to download the dataset. If that doesn't work on your machine, download [`h95.rda`](https://raw.githubusercontent.com/cran/phonTools/master/data/h95.rda) yourself and put it in the same folder as the notebook.

### Libraries used

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `matplotlib` | All plots |
| `scikit-learn` | K-Means, Adjusted Rand Index, scaling, train/validation/test split, confusion matrix |
| `torch` (PyTorch) | The neural network, following Tutorial 7 |
| `pyreadr` | Reading the vowel dataset, which is stored as an R data file |

## Exercise 1: K-Means clustering

**Data:** 20 customers (C1–C20), with annual income (k$) on the x-axis and spending score on the y-axis. All runs use `KMeans(n_init=10, random_state=42)` unless stated otherwise.

### Task 1.1: k = 3 vs k = 5

| k | Cluster | Centre (income, score) | Customers |
|---|---|---|---|
| 3 | Low spenders | (23.4, 12.8) | C3, C7, C9, C12, C16 |
| 3 | Medium spenders | (25.8, 39.8) | C1, C5, C13, C14, C18, C20 |
| 3 | High spenders | (24.2, 81.4) | C2, C4, C6, C8, C10, C11, C15, C17, C19 |
| 5 | Low spenders | (21.8, 11.0) | C3, C7, C9, C12 |
| 5 | Higher income, spend little | (32.0, 28.3) | C16, C18, C20 |
| 5 | Medium spenders | (22.3, 43.5) | C1, C5, C13, C14 |
| 5 | Big spenders, lower income | (19.5, 76.8) | C2, C4, C6, C10 |
| 5 | Big spenders, higher income | (28.0, 85.2) | C8, C11, C15, C17, C19 |

- **k = 3** (WCSS = 1,495) splits customers into three bands by spending. Income hardly matters, partly because spending scores range from 6 to 94 while incomes only range from 15 to 34, so a gap in spending counts for much more in the distance calculation.
- **k = 5** (WCSS = 852) splits the high spenders by income and gives C16, C18 and C20 their own group.

**Finding:** a lower WCSS doesn't automatically mean a better clustering. WCSS always goes down when you add more clusters, and would reach 0 if every customer had their own cluster.

### Task 1.2: Elbow method

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| WCSS | 17,886 | 3,306 | 1,495 | 1,171 | 852 | 635 |
| Drop from previous k | – | 82% | 55% | 22% | 27% | 25% |

The curve falls steeply until k = 3 and then flattens, so **the best number of clusters is k = 3** (low, medium and high spenders).

### Task 1.3: Sensitivity to initialisation

K-Means was run 10 times with a single random start each (`init="random", n_init=1`, seeds 0–9). Each result was compared with the best clustering found, using the Adjusted Rand Index (ARI; 1 = identical).

| k | Runs that found the best clustering | Different answers | WCSS range |
|---|---|---|---|
| 3 | 7 / 10 | 3 | 1,495 – 2,914 |
| 5 | 0 / 10 | 9 | 876 – 1,237 (best possible: 852) |

**Why:** each K-Means step can only lower WCSS, and the algorithm stops as soon as no step helps. So it finds the best answer *near where it started* (a local minimum). With more clusters there are more ways to start badly. In one k = 5 run, two centres got stuck on single customers while all the high spenders were squashed into one group. **Fixes:** k-means++ initialisation, many restarts keeping the lowest WCSS (`n_init`), and a fixed random seed.

## Exercise 2: Why K-Means fails on rings

**Data:** 30 points forming an inner ring (label 1, distance 0.44–0.73 from the centre) inside an outer ring (label 0, distance 0.95–1.03).

| Method | Accuracy |
|---|---|
| K-Means, k = 2, on (Feature1, Feature2) | **57%** (50–63% across 50 random starts) |
| K-Means, k = 2, on distance from the centre r = √(F1² + F2²) | **100%** |

**Why it fails:** K-Means assumes every cluster is a **round blob**. It describes each cluster with one centre point and gives every point to the closest centre. That breaks for rings:

- **The centre of a ring is empty,** and both rings share the same centre, so their centroids can't tell them apart.
- **The border between two K-Means clusters is always a straight line,** halfway between the centroids, and no straight line can separate a ring from the ring around it.

So K-Means cut the plane in half instead, and each half held pieces of both rings. Describing each point by its distance from the centre turns the rings into two separate blobs, which K-Means then separates perfectly.

## Exercise 3: Neural network vowel classification

### 3.1 Problem definition and motivation

**Task (binary classification):** given acoustic measurements of one vowel, decide whether the speaker said the vowel in ***head*** (/ɛ/, label 0) or ***had*** (/æ/, label 1).

These two vowels are easy to confuse. Many English learners find them hard, and many Singapore English speakers don't distinguish them at all. Telling them apart automatically is useful for speech recognition, pronunciation apps and speech therapy. No single measurement separates them, so the model has to combine several clues, which a neural network can learn.

### 3.2 Dataset and exploration

**Source:** Hillenbrand, Getty, Clark & Wheeler (1995), *Acoustic characteristics of American English vowels*, loaded from the `h95` dataset in the R package [phonTools](https://cran.r-project.org/package=phonTools). Only *head* and *had* were kept: **278 recordings** (139 each) from 139 speakers (45 men, 48 women, 27 boys, 19 girls).

| Input | Description |
|---|---|
| `dur` | How long the vowel lasts (ms) |
| `f0` | Pitch of the voice (Hz) |
| `f1` | First formant: related to tongue height (higher tongue → lower F1) |
| `f2` | Second formant: related to tongue frontness (further forward → higher F2) |
| `f3` | Third formant: mostly reflects the speaker's vocal-tract size |

The vowel codes in the raw data are X-SAMPA: `E` = /ɛ/ (*head*), `{` = /æ/ (*had*).

**Observations:**

- **Length is the strongest clue.** *Had* is about 80 ms longer on average (310 vs 230 ms; correlation with the label 0.59).
- **F1 and F2 overlap heavily.** Textbooks predict a higher F1 for *had*, but here it is slightly lower (663 vs 686 Hz). Most speakers are from Michigan, where the Northern Cities Shift raises the *had* vowel towards *head*.
- **Pitch and F3 say nothing about the word on their own** (correlation 0.00), but they tell the model what kind of voice it is hearing.

**Preprocessing:** there were no missing values. The inputs were standardised with `StandardScaler` (fitted on the training set only), because formants are in the thousands of Hz while duration is in the hundreds of ms. The data was split **60 / 20 / 20** into training, validation and test sets, using the same `train_test_split` calls as Tutorial 7 (`random_state=42`): 166 / 56 / 56 recordings.

### 3.3 Model design

The same `ANN` class as Tutorial 7, with 5 inputs instead of 2:

```
5 inputs (dur, f0, F1, F2, F3)  →  4 hidden neurons (ReLU)  →  1 output neuron (sigmoid)
```

The output is the probability that the word is *had*. A value of 0.5 or more is classified as *had*. The network has **29 parameters** (5 × 4 + 4 biases = 24, plus 4 × 1 + 1 bias = 5). A small network suits a small dataset of 166 training recordings.

| Setting | Value |
|---|---|
| Loss | `BCELoss` (binary cross-entropy) |
| Optimiser | Adam, learning rate 0.01 |
| Epochs | 500, full batch |

### 3.4–3.5 Training, validation and evaluation

| | Training | Validation | Test |
|---|---|---|---|
| Accuracy | 91.6% | 85.7% | **85.7%** |
| Loss at epoch 500 | 0.19 | 0.52 | 0.29 |

**Test confusion matrix:**

| | Predicted *head* | Predicted *had* |
|---|---|---|
| **Actual *head*** | 20 | 4 |
| **Actual *had*** | 4 | 28 |

The model gets 48 of 56 unseen recordings right, against 50% for guessing. It makes four mistakes in each direction, so it is not biased towards one word. With only 56 test recordings, one recording changes accuracy by almost 2 percentage points.

### 3.6 Overfitting study

**Mild overfitting.** All three losses fall together until **epoch 59**. After that, training loss keeps falling (0.29 → 0.19) while validation loss rises (0.36 → 0.52). Test loss barely changes (0.31 → 0.29), and validation accuracy stays at about 85%. The model is **not underfitting**, since training accuracy is above 90%.

**Remedies:** early stopping at epoch 59 (test accuracy **87.5%**), dropout or weight decay, keeping the network small, and more data.

### 3.7 Ablation study

**Learning rate** (4 hidden neurons, 500 epochs):

| Learning rate | Best epoch (lowest val. loss) | Train / val / test accuracy | Behaviour |
|---|---|---|---|
| 0.1 | 7 | 94.0% / 87.5% / 80.4% | Learns fast, then overfits. Validation loss jumps up and wobbles |
| **0.01** | 59 | 91.6% / 85.7% / **85.7%** | Fast and smooth. Best test score |
| 0.001 | 500 | 88.6% / 82.1% / 83.9% | Still learning at epoch 500 (slight underfitting) |

**Hidden layer size** (learning rate 0.01):

| Hidden neurons | Train / val / test accuracy |
|---|---|
| 2 | 89.8% / 83.9% / 87.5% |
| **4 (main)** | 91.6% / 85.7% / 85.7% |
| 16 | 98.8% / 85.7% / 80.4% |
| 64 | 100% / 87.5% / 80.4% |

Bigger networks memorise the training set: training accuracy goes up while test accuracy goes down.

**Inputs:**

| Inputs | Test accuracy |
|---|---|
| F1 + F2 only (2 inputs, like Tutorial 7) | 67.9% |
| + duration | 78.6% |
| All five | 85.7% |

With only F1 and F2, the decision boundary (plotted in the notebook) cuts through both groups, because the vowels overlap so much. Vowel length adds the most.

### 3.8 Case study

| Speaker | Real → predicted | Confidence | Duration (ms) |
|---|---|---|---|
| Boy 24 | *had* → *had* ✓ | 100% | 451 |
| Boy 8 | *head* → *head* ✓ | 100% | 193 |
| Man 56 | *head* → *had* ✗ | 94% | 261 |
| Man 78 | *head* → *had* ✗ | 93% | 244 |
| Man 47 | *head* → *had* ✗ | 83% | 266 |
| Man 78 | *had* → *had* ✓ | 100% | 326 |

Test accuracy by speaker group: men 76%, women 94%, boys 92%, girls 82%.

- **Successes** are clearly long or clearly short vowels.
- **5 of the 8 mistakes came from men,** and the three most confident mistakes were all men saying *head* slowly. Length is the strongest clue, so a long *head* reads as *had*.
- **Man 78** said both vowels with very similar formants (F2 ≈ 1,730 vs 1,840 Hz), and they differed mainly in length. The model never compares a speaker's two words, so it can't use that contrast.
- **The remaining mistakes** were made at only about 60% confidence, which makes them borderline cases.

The failures come from **insufficient features** (one length and one formant snapshot per recording) and **real overlap** in this accent, not from noisy data.

### 3.9 Discussion and future work

- **Improvements:** speaker normalisation (for example Lobanov), so each vowel is compared to the speaker's own voice; formant movement across the vowel, not just one snapshot; and splitting the data by speaker, so the model is tested only on new voices.
- **With more time or computing power:** all 12 vowels in the dataset, cross-validation for more reliable scores than one 56-recording test set, and learning directly from the audio (spectrograms).
- **Extensions:** testing on other accents, especially Singapore English, where these two vowels often merge, and use in pronunciation-practice or speech-therapy tools.

## Debugging notes

**1. A broken line at the end of the notebook.** An earlier version of the last cell contained:

```python
json.dump(res, open("results_ex3.json", "w"), indent=1, def ault=str)
```

The stray space in `def ault` stops **Run all** with:

```
SyntaxError: invalid syntax
```

`default` must be written as one word. This cell was later replaced when Exercise 3 was rewritten.

**2. A confusing vowel code.** The raw data labels *had* as `{`, which looked like a mistake. It is actually the [X-SAMPA](https://en.wikipedia.org/wiki/X-SAMPA) code for /æ/. X-SAMPA writes IPA symbols with plain keyboard characters, and older phonetics datasets use it. The notebook maps the codes to words (`E` → *head*, `{` → *had*) so the tables are readable.

## Use of Generative AI

As the assignment's "Learn with AI" requirement asks, I used generative AI as a study partner rather than an answer key. The exact prompts and my own summaries of the responses are logged in the PDF report.

| Area | How AI helped |
|---|---|
| **Concepts** | Explained why K-Means assumes round clusters, and why unscaled features let spending score dominate the distances in Exercise 1. |
| **Scoping the problem** | Helped me narrow Exercise 3 from all 12 vowels down to one confusable pair, so it could reuse the binary Tutorial 7 network instead of a more complex model. |
| **Code** | Helped adapt the Tutorial 7 `ANN` class and training loop to five inputs, and write the plotting and ablation code. |
| **Interpreting results** | Helped me read the loss curves, the confusion matrix and the learning-rate comparison in plain language. |

**Verification:** I didn't accept AI output blindly. For example, an early description said F1 rises "when the mouth is more open". From my phonetics background, I corrected this to the more precise explanation in terms of tongue height (a higher tongue gives a lower F1). I also checked that each figure matched what the explanation claimed, for example that the learning-rate plot did not show test loss.

## Key takeaways

- **K-Means is simple but has strong assumptions.** It works for round, well-separated groups. It depends on feature scales and starting points, and it fails on shapes like rings.
- **Choosing k is a judgement call.** The elbow method and interpretability pointed to the same answer, k = 3.
- **For a small dataset, a small network is enough.** The Tutorial 7 network with four hidden neurons outperformed bigger ones on unseen data.
- **Domain knowledge matters.** Knowing about vowel length, formants and the Northern Cities Shift explained both what the model learned and where it went wrong.

## Submission links

- **Report (PDF):** submitted via NTULearn
- **Video presentation:** [link](https://youtu.be/P-OVypJ-iuo)]
- **Notebook:** SC5002_Lab2.ipynb (in this repository)
