# Disaster Tweets Classification

A Natural Language Processing project for classifying crisis-related tweets into 12 disaster categories using classical machine learning, deep sequential neural networks, and BERT.

**Course:** CSE440 - Natural Language Processing (Summer 2026)  
**Student:** Avishek Biswas (ID: 23201427)  
**Section:** 03  
**Institution:** BRAC University  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb)

---

## 1. Project Overview

During natural and human-made emergencies, people frequently post real-time updates and emergency calls on social media platforms like Twitter. However, the sheer volume of posts makes it difficult for relief organizations and emergency services to find and filter actionable information quickly.

The objective of this project is to build and evaluate text classification models that can categorize tweets into **12 distinct disaster types**:
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

---

## 2. Dataset

- **Source:** CrisisNLP / CrisisBench dataset (`disaster_tweets_10k_1.csv`)
- **Total Samples:** 11,015 tweets
- **Class Balance:** 12 classes with ~800 to 1,080 samples each (Fire has 388 samples).
- **Split:** 70% Training (7,709 samples), 15% Validation (1,653 samples), and 15% Test (1,653 samples) using stratified sampling.

---

## 3. Implementation Workflow

### Step 1: Preprocessing & Cleaning
- Removed URLs, Twitter handles (`@user`), and HTML entities (`&amp;`, `&lt;`).
- Expanded common English contractions (`can't` -> `cannot`, `it's` -> `it is`).
- Preserved hashtag text while stripping the `#` symbol.
- Converted text to lowercase and removed punctuation and numbers.
- Tokenized text with NLTK `word_tokenize`, filtered English stopwords, and applied `WordNetLemmatizer`.

### Step 2: Feature Representations
- **TF-IDF:** Unigram and bigram TF-IDF with sublinear term frequency scaling (top 10,000 features) for classical ML models.
- **Word2Vec:** Trained 100-dimensional Word2Vec embeddings (Skip-Gram & CBOW) on the training corpus.
- **Keras Sequences:** Tokenized sequences with a vocabulary size of 15,000 and padded to a maximum length of 50 tokens.
- **BERT Tokenizer:** Subword WordPiece tokenization for transformer fine-tuning.

### Step 3: Models & Hyperparameter Tuning
I implemented and compared 10 different models, experimenting with 3 distinct hyperparameter configurations per model (30 total runs):
- **Classical Models:** Logistic Regression, Random Forest, Multinomial Naive Bayes.
- **Recurrent Neural Networks:** SimpleRNN, Bidirectional SimpleRNN, GRU, Bidirectional GRU, LSTM, Bidirectional LSTM.
- **Transformer:** Pretrained BERT Base (`google-bert/bert-base-uncased`) fine-tuned with AdamW and linear warmup.

---

## 4. Test Set Results

Below is the evaluation summary on the held-out test set (1,653 samples) for the best configuration of each model:

| Model | Representation | Best Configuration | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Soft-Voting Ensemble (Bonus)** | BERT + BiLSTM + LogReg | Weights (0.50, 0.30, 0.20) | **95.28%** | **0.9540** | **0.9515** | **0.9526** |
| **BERT Base** | Subword Tokens | LR = 3e-5, Batch = 32, Epochs = 4 | **94.86%** | **0.9490** | **0.9472** | **0.9479** |
| **Bidirectional LSTM** | Word2Vec (100d) | 128 units, Dropout = 0.2, Adam | **89.53%** | **0.8971** | **0.8938** | **0.8950** |
| **Bidirectional GRU** | Word2Vec (100d) | 128 units, Dropout = 0.2, Adam | **89.17%** | **0.8934** | **0.8905** | **0.8916** |
| **LSTM** | Word2Vec (100d) | 128 units, Dropout = 0.2, Adam | **88.02%** | **0.8819** | **0.8786** | **0.8798** |
| **GRU** | Word2Vec (100d) | 128 units, Dropout = 0.2, Adam | **87.66%** | **0.8785** | **0.8749** | **0.8762** |
| **Bidirectional SimpleRNN** | Word2Vec (100d) | 128 units, Dropout = 0.2, Adam | **83.18%** | **0.8350** | **0.8295** | **0.8317** |
| **Logistic Regression** | TF-IDF (1-2 ngrams) | C = 1.0, Balanced class weights | **82.88%** | **0.8312** | **0.8260** | **0.8279** |
| **Random Forest** | TF-IDF (1-2 ngrams) | n_estimators = 300, min_samples_split = 4 | **79.43%** | **0.8015** | **0.7890** | **0.7928** |
| **Multinomial Naive Bayes** | TF-IDF (1-2 ngrams) | alpha = 0.1 | **78.65%** | **0.7920** | **0.7812** | **0.7845** |
| **SimpleRNN** | Word2Vec (100d) | 64 units, Adam LR = 0.001 | **72.41%** | **0.7305** | **0.7188** | **0.7224** |

### Summary of Findings
1. **BERT Base** achieved the highest standalone performance (94.79% Macro F1), demonstrating the power of contextual bidirectional self-attention on social media language.
2. **Gated models (BiLSTM and BiGRU)** performed significantly better than vanilla SimpleRNN (89.5% vs 72.4%) because gating prevents vanishing gradients over tweet sequences.
3. **Logistic Regression** was the strongest classical baseline (82.79% Macro F1), training in less than a second while remaining competitive.
4. **Soft-Voting Ensemble** produced the overall best result (95.26% Macro F1), showing that combining transformer representations with recurrent and n-gram models helps correct individual edge cases.

---

## 5. Bonus Implementations

- **Soft-Voting Ensemble:** Combined predicted probability distributions from BERT Base, BiLSTM, and Logistic Regression with fixed weighting ($0.50 \cdot \text{BERT} + 0.30 \cdot \text{BiLSTM} + 0.20 \cdot \text{LogReg}$).
- **Interactive Web App (`app.py`):** Built a Streamlit interface that takes raw tweet text, predicts the disaster category, displays confidence scores for the top 3 classes, and provides emergency routing advice.
- **Ablation Studies:** Evaluated the performance trade-offs across representations (TF-IDF vs Word2Vec vs BERT) and cleaning techniques in the project report.

---

## 6. How to Run

### Running in Google Colab (Recommended)
1. Click the **Open in Colab** badge above or open [`disaster_nlp_classification.ipynb`](disaster_nlp_classification.ipynb).
2. Go to **Runtime -> Change runtime type** and select **T4 GPU**.
3. Click **Runtime -> Run all**. The dataset will be loaded directly from the GitHub repository.

### Running Locally
```bash
# Clone the repository
git clone https://github.com/xer0Xavishek/disaster-nlp-classification.git
cd disaster-nlp-classification

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py
```

---

## 7. Project Structure

```text
disaster-nlp-classification/
├── disaster_nlp_classification.ipynb   # Master notebook with complete code and outputs
├── disaster_tweets_10k_1.csv            # CrisisNLP dataset
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── app.py                               # Streamlit web application
└── report/                              # ACL-style research paper
    ├── report.tex                       # LaTeX source code
    └── custom.bib                       # References
```
