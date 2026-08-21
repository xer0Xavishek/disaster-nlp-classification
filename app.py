import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import string
import time

# Page configuration
st.set_page_config(
    page_title="Disaster Tweet Classification | CSE440",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Clean CSS (No AI-style gradients or glassmorphism, clean technical UI)
st.markdown("""
<style>
    /* Clean System Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1f2937;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Technical Header */
    .app-header {
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Clean Card Container */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Result Box */
    .prediction-box {
        background: #f9fafb;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 1rem 1.25rem;
    }

    .prediction-label {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
    }

    .confidence-text {
        font-size: 1rem;
        font-weight: 600;
        color: #2563eb;
    }

    .dispatch-text {
        font-size: 0.9rem;
        color: #4b5563;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px dashed #e5e7eb;
    }

    /* Technical Token Box */
    .token-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.85rem;
        background: #f3f4f6;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        color: #1f2937;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# Setup NLTK
@st.cache_resource(show_spinner=False)
def init_nltk():
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
        nltk.download(pkg, quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

init_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Category dispatch mapping
DISPATCH_ROUTING = {
    'Earthquake': 'Urban Search and Rescue (USAR), structural safety assessment, canine units.',
    'Flood': 'Swift-water rescue teams, inflatable boat units, high-ground evacuation.',
    'Wildfire': 'Forestry firefighting crews, aerial water drops, perimeter evacuation.',
    'Typhoon': 'Coastal surge monitoring, emergency storm shelters, power grid restoration.',
    'Transportation Accident': 'Highway patrol, heavy vehicle extrication, paramedic ambulances.',
    'Explosion': 'Hazardous materials (HazMat) inspection, blast injury trauma units, fire control.',
    'Shooting': 'Law enforcement tactical response, active threat containment, trauma triage.',
    'Bombing': 'Explosive Ordnance Disposal (EOD), forensic containment, mass casualty protocol.',
    'Haze': 'Public air quality advisories, respirator distribution, vulnerable group alerts.',
    'Meteor': 'Astronomical and civil defense verification, ground impact assessment.',
    'Building Collapse': 'Heavy structural search teams, crane extrication, acoustic listening devices.',
    'Fire': 'Municipal ladder trucks, high-volume pumper engines, structural fire suppression.'
}

# Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "that's": "that is", "there's": "there is", "'re": " are", "'ve": " have"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words and len(w) > 1]
    return " ".join(filtered)

# Load dataset and train models
@st.cache_resource(show_spinner=False)
def load_models():
    dataset_url = "https://raw.githubusercontent.com/xer0Xavishek/disaster-nlp-classification/refs/heads/main/disaster_tweets_10k_1.csv"
    try:
        df = pd.read_csv(dataset_url)
    except Exception:
        df = pd.read_csv('disaster_tweets_10k_1.csv')
        
    df['cleaned'] = df['tweet_text'].apply(clean_text)
    df = df[df['cleaned'].str.strip().str.len() > 0].reset_index(drop=True)
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['disaster_type']
    
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X, y)
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    classes = sorted(y.unique())
    return tfidf, lr, rf, classes

tfidf, lr_model, rf_model, class_names = load_models()

# App Header
st.markdown("""
<div class="app-header">
    <div class="app-title">Crisis Tweet Multi-Class Classifier</div>
    <div class="app-subtitle">CSE440: Natural Language Processing · Summer 2026 · BRAC University · Group 05</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Prediction Tool", "Test Set Evaluation", "Dataset Summary", "Project Information"])

with tab1:
    col_left, col_right = st.columns([3, 2], gap="medium")
    
    with col_left:
        st.markdown("**Select a sample text or enter custom input:**")
        
        sample_options = {
            "Custom Input": "",
            "Sample 1 (Earthquake)": "URGENT: 6.8 magnitude earthquake struck offshore, strong shaking felt across the province for 45 seconds, buildings damaged.",
            "Sample 2 (Flood)": "Water level reaching over 2 meters in downtown residential area, families stranded on rooftops waiting for rescue boats.",
            "Sample 3 (Wildfire)": "Forest fire spreading rapidly due to dry winds, mandatory evacuation order issued for residents in county zone 4.",
            "Sample 4 (Industrial Explosion)": "Loud blast reported at chemical storage facility, black smoke visible from several kilometers away, emergency crews responding.",
            "Sample 5 (Transportation Accident)": "Major multi-vehicle collision and overturned fuel tanker blocking interstate highway, paramedics on scene.",
            "Sample 6 (Shooting)": "Active shooter incident reported near market district, police officers securing the area, avoid vicinity."
        }
        
        chosen_sample = st.selectbox("Load sample incident:", list(sample_options.keys()), index=1)
        
        default_val = sample_options[chosen_sample] if chosen_sample != "Custom Input" else ""
        tweet_input = st.text_area("Input Tweet Text:", value=default_val, height=120, placeholder="Paste or type tweet content here...")
        
        col_m, col_b = st.columns([2, 1])
        with col_m:
            model_choice = st.selectbox("Inference Model:", ["Ensemble (Logistic Regression + Random Forest)", "Logistic Regression (TF-IDF)", "Random Forest (TF-IDF)"])
        with col_b:
            st.write("")
            st.write("")
            run_btn = st.button("Classify Tweet", type="primary", use_container_width=True)

    with col_right:
        st.markdown("**Classification Output:**")
        
        if tweet_input.strip():
            start_time = time.time()
            cleaned_str = clean_text(tweet_input)
            
            if not cleaned_str:
                st.warning("Input contains only stopwords or filtered characters. Please provide meaningful text.")
            else:
                x_vec = tfidf.transform([cleaned_str])
                p_lr = lr_model.predict_proba(x_vec)[0]
                p_rf = rf_model.predict_proba(x_vec)[0]
                
                if model_choice == "Logistic Regression (TF-IDF)":
                    probs = p_lr
                elif model_choice == "Random Forest (TF-IDF)":
                    probs = p_rf
                else:
                    probs = 0.60 * p_lr + 0.40 * p_rf
                
                elapsed_ms = (time.time() - start_time) * 1000
                pred_idx = np.argmax(probs)
                predicted_category = class_names[pred_idx]
                confidence_score = probs[pred_idx] * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 0.8rem; color: #6b7280; text-transform: uppercase;">Predicted Incident Category</div>
                    <div class="prediction-label">{predicted_category}</div>
                    <div class="confidence-text">{confidence_score:.2f}% Confidence</div>
                    <div class="dispatch-text">
                        <strong>Dispatch Action:</strong> {DISPATCH_ROUTING.get(predicted_category, 'Standard emergency response.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"Inference Latency: {elapsed_ms:.1f} ms | Model: {model_choice.split(' ')[0]}")
                
                st.markdown("**Top Class Probabilities:**")
                top_indices = np.argsort(probs)[::-1][:4]
                for idx in top_indices:
                    c_name = class_names[idx]
                    c_val = probs[idx]
                    st.write(f"**{c_name}**: {c_val * 100:.1f}%")
                    st.progress(float(c_val))
                
                st.markdown("**Normalized Tokens:**")
                st.markdown(f"<div class='token-box'>{cleaned_str.split()}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter tweet text or select a sample on the left to run classification.")

with tab2:
    st.markdown("### Test Set Performance (Held-Out Test Set, n = 1,647)")
    st.markdown("Evaluation results for the top configuration of each model family evaluated during the project:")
    
    results_table = pd.DataFrame([
        {"Model Family": "Soft-Voting Ensemble", "Representation": "BERT + BiLSTM + LogReg", "Configuration": "Weights (0.50, 0.30, 0.20)", "Test Accuracy": "99.88%", "Macro Precision": "0.9988", "Macro Recall": "0.9989", "Macro F1": "0.9989"},
        {"Model Family": "BERT Base", "Representation": "WordPiece Subwords", "Configuration": "Config 1 (LR=2e-5, Batch=32, Epochs=3)", "Test Accuracy": "99.82%", "Macro Precision": "0.9983", "Macro Recall": "0.9983", "Macro F1": "0.9983"},
        {"Model Family": "Random Forest", "Representation": "TF-IDF (1-2 ngrams)", "Configuration": "Config 3 (n_estimators=300, min_split=4)", "Test Accuracy": "98.85%", "Macro Precision": "0.9887", "Macro Recall": "0.9895", "Macro F1": "0.9891"},
        {"Model Family": "Logistic Regression", "Representation": "TF-IDF (1-2 ngrams)", "Configuration": "Config 2 (C=1.0, Balanced)", "Test Accuracy": "98.60%", "Macro Precision": "0.9876", "Macro Recall": "0.9874", "Macro F1": "0.9874"},
        {"Model Family": "Bidirectional GRU", "Representation": "Word2Vec (100d)", "Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "98.06%", "Macro Precision": "0.9830", "Macro Recall": "0.9824", "Macro F1": "0.9824"},
        {"Model Family": "Multinomial Naive Bayes", "Representation": "TF-IDF (1-2 ngrams)", "Configuration": "Config 3 (alpha=1.0)", "Test Accuracy": "97.27%", "Macro Precision": "0.9754", "Macro Recall": "0.9727", "Macro F1": "0.9738"},
        {"Model Family": "Bidirectional SimpleRNN", "Representation": "Word2Vec (100d)", "Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "95.63%", "Macro Precision": "0.9592", "Macro Recall": "0.9576", "Macro F1": "0.9576"},
        {"Model Family": "Bidirectional LSTM", "Representation": "Word2Vec (100d)", "Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "95.20%", "Macro Precision": "0.9547", "Macro Recall": "0.9556", "Macro F1": "0.9556"},
        {"Model Family": "SimpleRNN", "Representation": "Word2Vec (100d)", "Configuration": "Config 3 (2-Layer Stacked)", "Test Accuracy": "94.41%", "Macro Precision": "0.9482", "Macro Recall": "0.9466", "Macro F1": "0.9466"}
    ])
    st.dataframe(results_table, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Dataset Distribution")
    st.markdown("Total Records: **11,015** across **12 categories** (CrisisNLP benchmark):")
    
    cat_df = pd.DataFrame([
        {"Category": "Typhoon", "Sample Count": 1080, "Percentage": "9.81%"},
        {"Category": "Transportation Accident", "Sample Count": 1072, "Percentage": "9.73%"},
        {"Category": "Wildfire", "Sample Count": 1041, "Percentage": "9.45%"},
        {"Category": "Earthquake", "Sample Count": 1029, "Percentage": "9.34%"},
        {"Category": "Flood", "Sample Count": 1021, "Percentage": "9.27%"},
        {"Category": "Explosion", "Sample Count": 951, "Percentage": "8.63%"},
        {"Category": "Shooting", "Sample Count": 914, "Percentage": "8.30%"},
        {"Category": "Bombing", "Sample Count": 907, "Percentage": "8.23%"},
        {"Category": "Haze", "Sample Count": 903, "Percentage": "8.20%"},
        {"Category": "Meteor", "Sample Count": 877, "Percentage": "7.96%"},
        {"Category": "Building Collapse", "Sample Count": 832, "Percentage": "7.55%"},
        {"Category": "Fire", "Sample Count": 388, "Percentage": "3.52%"}
    ])
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Project & Team Details")
    st.markdown("""
    - **Course:** CSE440 - Natural Language Processing
    - **Semester:** Summer 2026
    - **Section:** 03
    - **Group:** 05
    - **Department:** Department of Computer Science and Engineering
    - **Institution:** BRAC University
    
    **Team Members:**
    1. **Avishek Biswas** — Student ID: 23201427
    2. **Sreema Roy** — Student ID: 23201444
    3. **Fahim Tasnim Khan** — Student ID: 23201087
    4. **Tawsif Kabir Pritom** — Student ID: 23201231
    
    **Project Links:**
    - GitHub Repository: [https://github.com/xer0Xavishek/disaster-nlp-classification](https://github.com/xer0Xavishek/disaster-nlp-classification)
    - Master Notebook: [disaster_nlp_classification.ipynb](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb)
    """)
