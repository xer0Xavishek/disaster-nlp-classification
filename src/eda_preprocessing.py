import os
import re
import html
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('disaster_tweets_10k_1.csv')
print(f"Dataset Loaded Successfully! Shape: {df.shape}")
print("\nColumn Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nUnique Disaster Types (Classes):", df['disaster_type'].nunique())
print(df['disaster_type'].value_counts())

# 2. Text Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Domain-specific words to keep or remove if needed
contractions_dict = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "she's": "she is", "that's": "that is", "what's": "what is",
    "there's": "there is", "'re": " are", "'ve": " have",
    "'ll": " will", "'d": " would"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # HTML decoding
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove User Mentions
    text = re.sub(r'@\w+', '', text)
    # Expand Contractions
    for cont, exp in contractions_dict.items():
        text = text.replace(cont, exp)
    # Remove Retweet markers and hashtags symbol
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Lowercasing
    text = text.lower()
    # Remove punctuation & numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal and lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 1
    ]
    return " ".join(cleaned_tokens)

print("\nExecuting Text Preprocessing Pipeline...")
df['cleaned_text'] = df['tweet_text'].apply(clean_text)
df['raw_char_len'] = df['tweet_text'].apply(len)
df['clean_char_len'] = df['cleaned_text'].apply(len)
df['raw_word_count'] = df['tweet_text'].apply(lambda x: len(str(x).split()))
df['clean_word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

# Filter out empty cleaned texts if any
initial_len = len(df)
df = df[df['cleaned_text'].str.strip().str.len() > 0].reset_index(drop=True)
print(f"Retained {len(df)} non-empty cleaned tweets from {initial_len} original samples.")

# 3. Visualizations & Statistical Summaries
# Figure 1: Class Distribution
plt.figure(figsize=(12, 6))
palette = sns.color_palette("viridis", df['disaster_type'].nunique())
ax = sns.countplot(data=df, y='disaster_type', order=df['disaster_type'].value_counts().index, palette=palette)
plt.title('Disaster Category Class Distribution (CrisisNLP/CrisisBench)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Count', fontsize=12)
plt.ylabel('Disaster Type', fontsize=12)
for p in ax.patches:
    ax.annotate(f"{int(p.get_width())}", (p.get_width() + 10, p.get_y() + p.get_height() / 2),
                va='center', fontsize=10)
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300)
plt.close()
print("Saved figures/class_distribution.png")

# Figure 2: Sequence Length Distributions
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['raw_char_len'], bins=30, color='skyblue', label='Raw Tweet Length', kde=True)
sns.histplot(df['clean_char_len'], bins=30, color='coral', label='Cleaned Tweet Length', kde=True)
plt.title('Character Length Distribution (Raw vs Cleaned)', fontsize=12, fontweight='bold')
plt.xlabel('Character Count')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(df['raw_word_count'], bins=25, color='teal', label='Raw Word Count', kde=True)
sns.histplot(df['clean_word_count'], bins=25, color='orange', label='Cleaned Word Count', kde=True)
plt.title('Word Count Distribution (Raw vs Cleaned)', fontsize=12, fontweight='bold')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('figures/length_distributions.png', dpi=300)
plt.close()
print("Saved figures/length_distributions.png")

# Figure 3: WordCloud for Top Disaster Types
top_classes = ['Earthquake', 'Flood', 'Wildfire', 'Typhoon']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, cat in zip(axes.flatten(), top_classes):
    cat_text = " ".join(df[df['disaster_type'] == cat]['cleaned_text'])
    wc = WordCloud(width=600, height=400, background_color='white', colormap='plasma', max_words=100).generate(cat_text)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud: {cat}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/wordclouds.png', dpi=300)
plt.close()
print("Saved figures/wordclouds.png")

# 4. Stratified Data Split (70% Train, 15% Val, 15% Test)
# Label Encoding
labels = sorted(df['disaster_type'].unique())
label2idx = {label: i for i, label in enumerate(labels)}
idx2label = {i: label for i, label in enumerate(labels)}
df['label_idx'] = df['disaster_type'].map(label2idx)

X = df['cleaned_text'].values
y = df['label_idx'].values
raw_X = df['tweet_text'].values

X_train_val, X_test, y_train_val, y_test, raw_train_val, raw_test = train_test_split(
    X, y, raw_X, test_size=0.15, random_state=42, stratify=y
)

# Split train_val into 70% Train (0.70 / 0.85 of train_val) and 15% Val (0.15 / 0.85 of train_val)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
    X_train_val, y_train_val, raw_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
)

print(f"\nStratified Data Split Complete:")
print(f"Training Set:   {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation Set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test Set:       {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# Save processed splits to CSV
train_df = pd.DataFrame({'tweet_text': raw_train, 'cleaned_text': X_train, 'label': y_train, 'disaster_type': [idx2label[i] for i in y_train]})
val_df = pd.DataFrame({'tweet_text': raw_val, 'cleaned_text': X_val, 'label': y_val, 'disaster_type': [idx2label[i] for i in y_val]})
test_df = pd.DataFrame({'tweet_text': raw_test, 'cleaned_text': X_test, 'label': y_test, 'disaster_type': [idx2label[i] for i in y_test]})

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
df.to_csv('data/full_preprocessed.csv', index=False)

print("Saved data splits into data/train.csv, data/val.csv, data/test.csv, and data/full_preprocessed.csv")
