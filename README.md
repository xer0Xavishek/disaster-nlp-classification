# Disaster Tweets Classification (CSE440 NLP Project)

**Student Name:** Avishek Biswas  
**Student ID:** 23201427  
**Course:** CSE440 - Natural Language Processing (Section 03)  
**Notebook Link:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb)

---

## About the Project

During disasters like earthquakes, floods, or wildfires, people post thousands of updates, distress calls, and damage reports on social media. For rescue teams and emergency responders, sorting through this flood of raw text manually takes too long.

In this project, I built an end-to-end NLP classification pipeline that categorizes crisis tweets into **12 disaster types**:
1. Earthquake
2. Flood
3. Wildfire
4. Typhoon
5. Transportation Accident
6. Explosion
7. Shooting
8. Bombing
9. Haze
10. Meteor
11. Building Collapse
12. Fire

The dataset contains 11,015 labeled tweets from CrisisNLP / CrisisBench.

---

## What I Did in This Project

### 1. Data Cleaning & Text Preprocessing
Social media text is very messy, so I implemented a cleaning pipeline:
- Removed URLs (`http...`), user mentions (`@user`), and HTML entities (`&amp;`).
- Expanded common contractions (`can't` -> `cannot`, `it's` -> `it is`).
- Kept the text from hashtags while stripping `#`.
- Converted all text to lowercase and removed special symbols and numbers.
- Tokenized text using NLTK `word_tokenize`, removed English stopwords, and applied `WordNetLemmatizer`.

### 2. Feature Extraction & Word Embeddings
- **TF-IDF**: Extracted unigrams and bigrams (up to 10,000 features) with sublinear term frequency scaling for classical ML models.
- **Word2Vec (Skip-Gram & CBOW)**: Trained 100-dimensional domain word embeddings directly on the training tweets to capture disaster-specific semantics.
- **Keras Sequences & Embedding Matrix**: Built a vocabulary of 15,000 tokens with sequence length truncated/padded to 50 tokens.
- **BERT Tokenizer**: WordPiece subword tokenization for transformer fine-tuning.

### 3. Model Building & Hyperparameter Tuning
I implemented and compared **10 different models**, testing at least 3 configurations for each (30 total runs) across a 70% Train / 15% Val / 15% Test stratified split:
- **Classical ML**: Logistic Regression, Random Forest, Multinomial Naive Bayes.
- **Sequential Deep Learning**: SimpleRNN, Bidirectional SimpleRNN, GRU, Bidirectional GRU, LSTM, Bidirectional LSTM.
- **Transformers**: Fine-tuned `google-bert/bert-base-uncased` with AdamW and linear learning rate warmup.

---

## Test Results & Comparison

Here are the test set results for the best configuration of each model:

| Model | Feature Representation | Best Hyperparameters | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Soft-Voting Ensemble (Bonus)** | BERT + BiLSTM + LogReg | Weights: $0.50, 0.30, 0.20$ | **95.28%** | **0.9526** | **0.9529** |
| **BERT Base** | Subword Tokens | LR = 3e-5, Batch = 32, Epochs = 4 | **94.86%** | **0.9479** | **0.9484** |
| **Bidirectional LSTM** | Word2Vec Skip-Gram (100d) | 128 units, Dropout = 0.2, Adam | **89.53%** | **0.8950** | **0.8952** |
| **Bidirectional GRU** | Word2Vec Skip-Gram (100d) | 128 units, Dropout = 0.2, Adam | **89.17%** | **0.8916** | **0.8919** |
| **LSTM** | Word2Vec Skip-Gram (100d) | 128 units, Dropout = 0.2, Adam | **88.02%** | **0.8798** | **0.8804** |
| **GRU** | Word2Vec Skip-Gram (100d) | 128 units, Dropout = 0.2, Adam | **87.66%** | **0.8762** | **0.8768** |
| **Bidirectional SimpleRNN** | Word2Vec Skip-Gram (100d) | 128 units, Dropout = 0.2, Adam | **83.18%** | **0.8317** | **0.8321** |
| **Logistic Regression** | TF-IDF (1-2 ngrams) | C = 1.0, Balanced class weights | **82.88%** | **0.8279** | **0.8285** |
| **Random Forest** | TF-IDF (1-2 ngrams) | n = 300, min_samples_split = 4 | **79.43%** | **0.7928** | **0.7940** |
| **Multinomial Naive Bayes** | TF-IDF (1-2 ngrams) | alpha = 0.1 | **78.65%** | **0.7845** | **0.7861** |
| **SimpleRNN** | Word2Vec Skip-Gram (100d) | 64 units, Adam LR = 0.001 | **72.41%** | **0.7224** | **0.7238** |

### Key Observations:
1. **BERT performed the best** because pre-trained bidirectional self-attention understands sentence context and slang much better than static embeddings.
2. **Bidirectional LSTM and GRU beat vanilla RNNs** by a large margin (89.5% vs 72.4%) because gating avoids vanishing gradients over sequence steps.
3. **Logistic Regression with TF-IDF** is very fast to train and surprisingly competitive (~83%), making it a great lightweight baseline.
4. **Soft-voting ensemble** gave the highest overall accuracy (95.28%) by combining BERT's contextual understanding with BiLSTM and n-gram keyword signals.

---

## Bonus Implementations

1. **Soft-Voting Ensemble Model**: Combined predicted probabilities from BERT Base (0.50), BiLSTM (0.30), and Logistic Regression (0.20) to boost test performance.
2. **Interactive Streamlit Web App (`app.py`)**: A simple web UI where you can paste any tweet and see the predicted disaster class, top 3 probability scores, and suggested emergency dispatch action.
3. **Ablation Studies**: Tested the impact of different representations (TF-IDF vs Word2Vec vs BERT) and cleaning steps in the final report.

---

## Repository Files

```text
disaster-nlp-classification/
├── disaster_nlp_classification.ipynb   # Master Google Colab notebook (Self-contained)
├── disaster_tweets_10k_1.csv            # 12-class dataset (11,015 samples)
├── requirements.txt                     # Python packages needed to run locally
├── README.md                            # Project documentation
├── viva_preparation_guide.md            # Detailed study notes and viva Q&A
├── app.py                               # Streamlit web app for live predictions
└── report/                              # 8-Page ACL format research paper
    ├── report.tex                       # LaTeX source code
    └── custom.bib                       # References bibliography
```

---

## How to Run

### In Google Colab (Recommended):
Open [`disaster_nlp_classification.ipynb`](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb), switch the runtime type to GPU (T4), and click **Runtime -> Run all**. The notebook automatically pulls the dataset from GitHub.

### Locally:
```bash
# Clone the repository
git clone https://github.com/xer0Xavishek/disaster-nlp-classification.git
cd disaster-nlp-classification

# Create virtual environment and install packages
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run app.py
```
