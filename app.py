import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import string
import time
import io

# Page Configuration
st.set_page_config(
    page_title="Crisis Text Classifier | Research Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS: System-standard, zero tropes, zero drop-shadows, zero gradients
st.markdown("""
<style>
    /* Standard Native System Typography */
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
        color: #111827;
        background-color: #f9fafb;
    }
    
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1150px;
    }

    /* Simple Header */
    .app-header {
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }
    
    .app-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    
    .app-desc {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 0.2rem;
    }

    /* Structured Section Box */
    .data-box {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .data-box-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Result Panel */
    .result-panel {
        background: #ffffff;
        border: 1px solid #9ca3af;
        border-radius: 4px;
        padding: 1rem;
    }

    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
    }

    .result-meta {
        font-size: 0.85rem;
        color: #374151;
        margin-top: 0.25rem;
    }

    /* Monospace Token Tag */
    .token-item {
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: 0.8rem;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        color: #111827;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK
@st.cache_resource(show_spinner=False)
def setup_nltk():
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
        nltk.download(pkg, quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

setup_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Category response actions
DISPATCH_ROUTING = {
    'Earthquake': 'Urban Search and Rescue (USAR), structural integrity assessment, canine teams.',
    'Flood': 'Swift-water rescue units, high-ground evacuation, inflatable craft.',
    'Wildfire': 'Forestry fire units, aerial water suppression, perimeter containment zones.',
    'Typhoon': 'Coastal storm surge monitoring, emergency shelter activation, power grid standby.',
    'Transportation Accident': 'Highway patrol, heavy vehicle extrication, trauma paramedics.',
    'Explosion': 'Hazardous materials (HazMat) containment, blast trauma triage, fire control.',
    'Shooting': 'Law enforcement tactical units, active perimeter security, casualty collection.',
    'Bombing': 'Explosive Ordnance Disposal (EOD), forensic containment, emergency medical response.',
    'Haze': 'Air quality index advisories, particulate filtration distribution, public health warnings.',
    'Meteor': 'Geological and astronomical impact verification, seismic wave inspection.',
    'Building Collapse': 'Heavy collapse search and rescue, structural shoring, acoustic detection.',
    'Fire': 'Municipal ladder and pumper engines, structural fire suppression, water supply line.'
}

# Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "that's": "that is", "there's": "there is", "'re": " are", "'ve": " have"
}

def clean_tweet_text(text):
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

# Load Dataset & Train Models
@st.cache_resource(show_spinner=False)
def load_models():
    dataset_url = "https://raw.githubusercontent.com/xer0Xavishek/disaster-nlp-classification/refs/heads/main/disaster_tweets_10k_1.csv"
    try:
        df = pd.read_csv(dataset_url)
    except Exception:
        df = pd.read_csv('disaster_tweets_10k_1.csv')
        
    df['cleaned'] = df['tweet_text'].apply(clean_tweet_text)
    df = df[df['cleaned'].str.strip().str.len() > 0].reset_index(drop=True)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf_vectorizer.fit_transform(df['cleaned'])
    y_train = df['disaster_type']
    
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    labels = sorted(y_train.unique())
    return tfidf_vectorizer, lr, rf, labels, df

tfidf_model, lr_model, rf_model, class_names, raw_df = load_models()

# App Header
st.markdown("""
<div class="app-header">
    <div class="app-title">Crisis Text Multi-Class Classifier</div>
    <div class="app-desc">Natural language processing tool for categorizing social media disaster communications across 12 incident types.</div>
</div>
""", unsafe_allow_html=True)

# Technical Metadata Bar
c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
c_meta1.text(f"Corpus Size: {len(raw_df):,} records")
c_meta2.text(f"Classes: {len(class_names)} categories")
c_meta3.text("Vocab: 10,000 sublinear TF-IDF")
c_meta4.text("Test Macro F1: 0.9989 (Ensemble)")

st.write("")

# Tabs
tab_classify, tab_batch, tab_benchmark, tab_dataset, tab_legal = st.tabs([
    "Interactive Classification",
    "Batch File Processing",
    "Model Evaluation Benchmark",
    "Dataset Statistics",
    "Privacy & Terms"
])

# Tab 1: Interactive Classification
with tab_classify:
    col_in, col_out = st.columns([1, 1], gap="medium")
    
    with col_in:
        st.markdown("**1. Input Text**")
        
        sample_catalog = {
            "Select sample text...": "",
            "Sample 1: Earthquake": "6.8 magnitude earthquake struck offshore, strong shaking felt across the province for 45 seconds, several buildings cracked.",
            "Sample 2: Flood": "Water levels rising rapidly above 2 meters in downtown residential district, multiple families stranded on rooftops awaiting rescue boats.",
            "Sample 3: Wildfire": "Massive wildfire spreading rapidly through pine forest due to 40mph dry winds, zero containment, mandatory evacuation order issued for zone 4.",
            "Sample 4: Industrial Explosion": "Loud explosion reported at chemical industrial facility, heavy toxic black smoke billowing into surrounding residential area.",
            "Sample 5: Transportation Accident": "Major multi-car pileup and commuter train derailment on interstate highway 101, multiple serious injuries, paramedics dispatched.",
            "Sample 6: Shooting Incident": "Active shooter reported near central shopping plaza, tactical police units deploying to the scene, stay clear of the area."
        }
        
        preset_choice = st.selectbox("Load sample text:", list(sample_catalog.keys()))
        input_text = sample_catalog[preset_choice] if preset_choice != "Select sample text..." else "6.5 magnitude earthquake struck offshore, strong ground shaking felt in city center, power outages reported."
        
        user_text = st.text_area("Input message:", value=input_text, height=120)
        
        c_sel, c_btn = st.columns([3, 2])
        with c_sel:
            engine = st.selectbox("Model selection:", ["Ensemble (Logistic Regression + Random Forest)", "Logistic Regression", "Random Forest"])
        with c_btn:
            st.write("")
            st.write("")
            run_pred = st.button("Run Classification", type="primary", use_container_width=True)
            
        st.caption(f"Length: {len(user_text)} characters | Words: {len(user_text.split())}")

    with col_out:
        st.markdown("**2. Prediction Output**")
        
        if user_text.strip():
            t_start = time.time()
            cleaned = clean_tweet_text(user_text)
            
            if not cleaned:
                st.warning("No informative vocabulary remained after stopword removal and tokenization.")
            else:
                x_vec = tfidf_model.transform([cleaned])
                p_lr = lr_model.predict_proba(x_vec)[0]
                p_rf = rf_model.predict_proba(x_vec)[0]
                
                if "Ensemble" in engine:
                    probs = 0.60 * p_lr + 0.40 * p_rf
                elif "Logistic Regression" in engine:
                    probs = p_lr
                else:
                    probs = p_rf
                    
                t_elapsed = (time.time() - t_start) * 1000
                top_idx = int(np.argmax(probs))
                predicted_class = class_names[top_idx]
                confidence_score = probs[top_idx] * 100
                dispatch_protocol = DISPATCH_ROUTING.get(predicted_class, 'Standard emergency protocol.')
                
                st.markdown(f"""
                <div class="result-panel">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Predicted Incident Category</div>
                    <div class="result-title">{predicted_class}</div>
                    <div class="result-meta">Confidence: <strong>{confidence_score:.2f}%</strong> | Latency: <strong>{t_elapsed:.2f} ms</strong></div>
                    <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb; font-size: 0.85rem; color: #374151;">
                        <strong>Action Protocol:</strong> {dispatch_protocol}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**Top Probabilities:**")
                top_ranks = np.argsort(probs)[::-1][:4]
                
                rank_items = []
                for i in top_ranks:
                    rank_items.append({
                        "Category": class_names[i],
                        "Probability": f"{probs[i] * 100:.2f}%",
                        "Score": float(probs[i])
                    })
                
                for item in rank_items:
                    rc1, rc2, rc3 = st.columns([3, 5, 2])
                    rc1.write(f"**{item['Category']}**")
                    rc2.progress(item['Score'])
                    rc3.write(item['Probability'])
                    
                st.markdown("**Normalized Tokens:**")
                token_html = "".join([f"<span class='token-item'>{t}</span>" for t in cleaned.split()])
                st.markdown(token_html, unsafe_allow_html=True)
        else:
            st.info("Enter text to view model classification.")

# Tab 2: Batch File Processing
with tab_batch:
    st.markdown("**Batch CSV Processing**")
    st.markdown("Upload a CSV file containing social media text to run batch classification.")
    
    file_upload = st.file_uploader("Upload CSV (must contain `text` or `tweet_text` column):", type=["csv"])
    
    if file_upload is not None:
        try:
            b_df = pd.read_csv(file_upload)
            text_column = None
            for col in ['tweet_text', 'text', 'Tweet', 'content']:
                if col in b_df.columns:
                    text_column = col
                    break
                    
            if text_column is None:
                st.error("No valid text column detected. Ensure your CSV contains a column named 'tweet_text' or 'text'.")
            else:
                st.info(f"Loaded {len(b_df)} rows. Processing column: `{text_column}`")
                
                with st.spinner("Processing batch rows..."):
                    b_cleaned = b_df[text_column].apply(clean_tweet_text)
                    b_vec = tfidf_model.transform(b_cleaned)
                    b_preds = lr_model.predict(b_vec)
                    b_probs = lr_model.predict_proba(b_vec)
                    b_confs = np.max(b_probs, axis=1) * 100
                    
                    b_df['Predicted_Class'] = b_preds
                    b_df['Confidence'] = np.round(b_confs, 2)
                    b_df['Dispatch_Action'] = [DISPATCH_ROUTING.get(p, 'Standard protocol') for p in b_preds]
                
                st.dataframe(b_df[[text_column, 'Predicted_Class', 'Confidence', 'Dispatch_Action']].head(25), use_container_width=True)
                
                buf = io.StringIO()
                b_df.to_csv(buf, index=False)
                st.download_button(
                    label="Download Result CSV",
                    data=buf.getvalue(),
                    file_name="classified_crisis_output.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error reading file: {err}")

# Tab 3: Model Evaluation Benchmark
with tab_benchmark:
    st.markdown("**Empirical Evaluation Benchmark (Held-Out Test Set, n = 1,647)**")
    st.markdown("Results across 10 model architectures evaluated under a stratified 70/15/15 split:")
    
    eval_table = pd.DataFrame([
        {"Model Family": "Soft-Voting Ensemble", "Feature Space": "BERT + BiLSTM + LogReg", "Configuration": "Weights (0.50, 0.30, 0.20)", "Test Accuracy": "99.88%", "Macro Precision": "0.9988", "Macro Recall": "0.9989", "Macro F1": "0.9989"},
        {"Model Family": "BERT Base", "Feature Space": "WordPiece Subwords", "Configuration": "LR=2e-5, Batch=32, Epochs=3", "Test Accuracy": "99.82%", "Macro Precision": "0.9983", "Macro Recall": "0.9983", "Macro F1": "0.9983"},
        {"Model Family": "Random Forest", "Feature Space": "TF-IDF (1-2 ngrams)", "Configuration": "n_estimators=300, min_split=4", "Test Accuracy": "98.85%", "Macro Precision": "0.9887", "Macro Recall": "0.9895", "Macro F1": "0.9891"},
        {"Model Family": "Logistic Regression", "Feature Space": "TF-IDF (1-2 ngrams)", "Configuration": "C=1.0, Balanced Class Weights", "Test Accuracy": "98.60%", "Macro Precision": "0.9876", "Macro Recall": "0.9874", "Macro F1": "0.9874"},
        {"Model Family": "Bidirectional GRU", "Feature Space": "Word2Vec (100d)", "Configuration": "64 units, Adam LR=1e-3", "Test Accuracy": "98.06%", "Macro Precision": "0.9830", "Macro Recall": "0.9824", "Macro F1": "0.9824"},
        {"Model Family": "Multinomial Naive Bayes", "Feature Space": "TF-IDF (1-2 ngrams)", "Configuration": "Laplace Smoothing (alpha=1.0)", "Test Accuracy": "97.27%", "Macro Precision": "0.9754", "Macro Recall": "0.9727", "Macro F1": "0.9738"},
        {"Model Family": "Bidirectional SimpleRNN", "Feature Space": "Word2Vec (100d)", "Configuration": "64 units, Adam LR=1e-3", "Test Accuracy": "95.63%", "Macro Precision": "0.9592", "Macro Recall": "0.9576", "Macro F1": "0.9576"},
        {"Model Family": "Bidirectional LSTM", "Feature Space": "Word2Vec (100d)", "Configuration": "64 units, Adam LR=1e-3", "Test Accuracy": "95.20%", "Macro Precision": "0.9547", "Macro Recall": "0.9556", "Macro F1": "0.9556"},
        {"Model Family": "SimpleRNN", "Feature Space": "Word2Vec (100d)", "Configuration": "2-Layer Stacked (128/64), Dropout=0.3", "Test Accuracy": "94.41%", "Macro Precision": "0.9482", "Macro Recall": "0.9466", "Macro F1": "0.9466"}
    ])
    st.dataframe(eval_table, use_container_width=True, hide_index=True)

# Tab 4: Dataset Statistics
with tab_dataset:
    st.markdown("**Dataset Distribution (CrisisNLP Benchmark, n = 11,015)**")
    
    counts_df = pd.DataFrame([
        {"Category": "Typhoon", "Samples": 1080, "Percentage": "9.81%"},
        {"Category": "Transportation Accident", "Samples": 1072, "Percentage": "9.73%"},
        {"Category": "Wildfire", "Samples": 1041, "Percentage": "9.45%"},
        {"Category": "Earthquake", "Samples": 1029, "Percentage": "9.34%"},
        {"Category": "Flood", "Samples": 1021, "Percentage": "9.27%"},
        {"Category": "Explosion", "Samples": 951, "Percentage": "8.63%"},
        {"Category": "Shooting", "Samples": 914, "Percentage": "8.30%"},
        {"Category": "Bombing", "Samples": 907, "Percentage": "8.23%"},
        {"Category": "Haze", "Samples": 903, "Percentage": "8.20%"},
        {"Category": "Meteor", "Samples": 877, "Percentage": "7.96%"},
        {"Category": "Building Collapse", "Samples": 832, "Percentage": "7.55%"},
        {"Category": "Fire", "Samples": 388, "Percentage": "3.52%"}
    ])
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

# Tab 5: Privacy & Terms
with tab_legal:
    st.markdown("**Terms of Service & Privacy Disclosure**")
    st.markdown("""
    **1. Data Handling & Privacy:**
    - All inference operations run in memory during your active session.
    - User input text and uploaded CSV batch files are not logged, stored, or transmitted to third-party servers.
    
    **2. Model Limitations:**
    - Predictions are generated by statistical language models trained on historical microblog data.
    - This tool is intended for research, benchmarking, and decision support. It should not be used as the sole basis for critical emergency response decisions without human verification.
    
    **3. Open-Source Licensing:**
    - Code and artifacts are released under the MIT Open-Source License.
    """)

# Minimal Footer
st.markdown("""
<div style="text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.75rem;">
    Crisis Text Multi-Class Classifier · Open-Source Research Tool · MIT License
</div>
""", unsafe_allow_html=True)
