# SC5002 Lab Assignment 1: Search Algorithms & Regularised Regression #

**Course:** SC5002 Artificial Intelligence Fundamentals and Applications, Nanyang Technological University
**Author:** Yeo Wan Ling
**Semester:** 2026 S1

This repository contains my work for Lab Assignment 1. It has two parts:

1. **Classical search:** tracing Breadth-First Search (BFS) and A* by hand on a weighted graph, and solving the 8-puzzle with A*
2. **Applied machine learning:** comparing Linear Regression and Ridge Regression on the California Housing dataset in Google Colab

Generative AI was used throughout as a learning partner, as the assignment requires. How I used it is described in [Use of Generative AI](#use-of-generative-ai).

---

## Contents

- [Repository structure](#repository-structure)
- [How to run the notebook](#how-to-run-the-notebook)
- [Part 1: Classical search](#part-1-classical-search)
- [Part 2: Machine learning](#part-2-machine-learning)
- [Debugging note](#debugging-note)
- [Use of Generative AI](#use-of-generative-ai)
- [Key takeaways](#key-takeaways)
- [Submission links](#submission-links)

---

## Repository structure

```
.
├── README.md                    # This file
└── [notebook_name].ipynb        # Completed Google Colab notebook (Part 2)
```

> The Part 1 hand calculations, the AI prompting logs and the score tables are in the PDF report submitted on NTULearn.

---

## How to run the notebook

### Option A: Google Colab (recommended)

1. Open the notebook in Colab: **[Open in Colab]([colab-link])**
2. Click **Runtime → Run all**.
3. No installation is needed. Colab already includes every library used.

### Option B: Run locally

You need Python 3.9 or later.

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook [notebook_name].ipynb
```

The first run downloads the California Housing dataset through `sklearn.datasets.fetch_california_housing`, so you need an internet connection.

### Libraries used

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `matplotlib` | Bar chart of model performance |
| `scikit-learn` | Dataset, scaling, models and cross-validation |

---

## Part 1: Classical search

Part 1 involved no code. All the working was done by hand, and the full step-by-step tables are in the PDF report. This is a summary of the results.

### Task 1.1: BFS vs A* on a weighted graph

**Problem:** a directed graph from start state **S** to goal **G**.

- Edges: S→A (2), S→B (1), S→F (3), A→C (2), A→D (3), B→D (2), B→E (4), C→G (4), D→G (4), F→G (6)
- Admissible heuristic: h(S)=6, h(A)=4, h(B)=5, h(C)=2, h(D)=2, h(E)=8, h(F)=4, h(G)=0

| Algorithm | Expansion order | Path found | Cost |
|---|---|---|---|
| **BFS** (FIFO queue, ignores costs) | S, A, B, F, C, D, E, G | S → F → G | 9 |
| **A\*** (lowest f(n) = g(n) + h(n)) | S, A, B, D, C, F, G | **S → B → D → G** | **7** |

**Finding:** BFS returns the *shallowest* path, the one with the fewest edges. A* returns the *cheapest* path. BFS is only optimal when every step costs the same. A* is optimal whenever the heuristic is admissible, meaning it never overestimates the true cost. For example, h(B) = 5 ≤ 6, the true cost from B to G.

Ties in A* were broken alphabetically.

### Task 1.2: 8-puzzle with A*

```
Start          Goal
2 8 3          1 2 3
1 6 4    →     8 _ 4
7 _ 5          7 6 5
```

- **g(n)** = depth of the node (the number of moves so far)
- **h(n)** = number of misplaced tiles, not counting the blank

| Step | Tile moved | State | g | h | f |
|---|---|---|---|---|---|
| 0 | – | `283 / 164 / 7_5` | 0 | 4 | 4 |
| 1 | 6 ↓ | `283 / 1_4 / 765` | 1 | 3 | 4 |
| 2 | 8 ↓ | `2_3 / 184 / 765` | 2 | 3 | 5 |
| 3 | 2 → | `_23 / 184 / 765` | 3 | 2 | 5 |
| 4 | 1 ↑ | `123 / _84 / 765` | 4 | 1 | 5 |
| 5 | 8 ← | `123 / 8_4 / 765` (goal) | 5 | 0 | 5 |

**Result:** the goal is reached in **5 moves**. f never exceeds 5 along the solution path.

**Lessons from the working:**
- When two states tie on f, A* compares across the *entire* OPEN list, not just the newest children. One of the tied branches turned out to be a dead end.
- A move that recreates an already-explored board is a **repeated state**. It should be marked as closed rather than expanded again, otherwise the search wastes effort or loops.

---

## Part 2: Machine learning

### Task 2.1: Dataset

The notebook loads the **California Housing** dataset from scikit-learn and takes a random sample of **1,000 rows** (`random_state=42`). Each row describes one district in California.

| Role | Column | Description |
|---|---|---|
| **Target** (continuous) | `MedHouseVal` | Median house value |
| Feature | `MedInc` | Median income |
| Feature | `HouseAge` | Median house age |
| Feature | `AveRooms` | Average rooms per household |
| Feature | `AveBedrms` | Average bedrooms per household |
| Feature | `Population` | District population |
| Feature | `AveOccup` | Average occupants per household |
| Feature | `Latitude` | How far north the district is |
| Feature | `Longitude` | How far east or west the district is |

**Preprocessing:** all features are standardised with `StandardScaler`. Without it, a feature measured in the thousands (`Population`) would dominate one measured around 1 (`AveBedrms`). The data is then split 80/20 into training and test sets.

### Task 2.2: Model training and alpha tuning

Two models were compared using **5-fold cross-validation** on the training set:

- **Linear Regression:** ordinary least squares, with no penalty
- **Ridge Regression:** Linear Regression plus an L2 penalty on large coefficients, controlled by `alpha` (α)

| Model | α | Mean 5-fold R² |
|---|---|---|
| Linear Regression | – | 0.6457 |
| Ridge Regression | 0.1 | 0.6457 |
| Ridge Regression | 10 | 0.6443 |
| Ridge Regression | 1000 | 0.3574 |

**Why cross-validation?** A single train/test split depends on which houses happen to land in the test set. With 5-fold cross-validation, each fold takes a turn as validation data, and the five scores are averaged. This gives a fairer estimate of how the model performs on unseen data.

### Task 2.3: Visual analysis and overfitting

The notebook plots the four mean R² scores as a bar chart.

- **Small α (0.1, 10):** almost no change from Linear Regression. With only eight features and 800 training rows, the linear model was not overfitting much, so the penalty had little to correct. α = 10 even scored very slightly lower.
- **Large α (1000):** R² drops to **0.357**. The penalty shrinks the coefficients so much that the model **underfits**. It can no longer capture real patterns, such as the link between income and house value.

**Takeaway:** regularisation is a balance. Too little penalty risks overfitting, which means memorising noise. Too much causes underfitting, which means missing real patterns. The right α captures genuine patterns without clinging to noise.

---

## Debugging note

The code template in the lab manual contained a line that had been wrapped across two lines to fit the page:

```python
print(f"Ridge Regression (alpha={a:6.1f})
    Mean 5-Fold R2 Score: {scores.mean():.4f}")
```

Pasting this into Colab raised:

```
SyntaxError: unterminated f-string literal
```

A normal Python string must close on the same line it opens on. The fix was to join it back into one line:

```python
print(f"Ridge Regression (alpha={a:6.1f}) Mean 5-Fold R2 Score: {scores.mean():.4f}")
```

The indentation lost when pasting from the PDF also had to be restored by hand.

---

## Use of Generative AI

As the assignment's "Learn with AI" requirement asks, I used generative AI as a study partner rather than an answer key. The exact prompts and my own summaries of the responses are logged in the PDF report.

| Area | How AI helped |
|---|---|
| **Concepts** | Explained admissible heuristics with a GPS analogy. A good heuristic is like a GPS distance estimate that never overshoots the real drive, so it never rules out the shortest route. |
| **Checking working** | Reviewed my A* tables. It caught a typo (f = 23 instead of 13), flagged that I hadn't stated the final path, and pointed out that repeated states in the 8-puzzle should be marked as closed. |
| **Debugging** | Explained the `unterminated f-string literal` error and why the string had to sit on one line. |
| **Interpreting results** | Helped me explain, in plain language, why Ridge behaved as it did and why cross-validation is more reliable than a single split. |

**Verification:** I didn't accept AI output blindly. Some answers depended on assumptions, such as how to break ties in A*. I checked these against my lecture slides and stated them explicitly in my working.

---

## Key takeaways

The common thread across the assignment is **balance**:

- **A\*** balances the cost so far, g(n), against the estimated cost remaining, h(n).
- **Ridge Regression** balances fitting the data closely against keeping the model simple.
- **Generative AI** was most useful when trusted enough to learn from, but still checked.

---

## Submission links

- **Report (PDF):** submitted via NTULearn
- **Video presentation:** [(https://youtu.be/G4lSrIAJcLc)]
- **Notebook:** SC5002_Lab1.ipynb (in this repository)
