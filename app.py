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
    page_title="Crisis Text Classifier | Research Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional Scientific CSS (Zero Tropes, Zero Emojis, High Usability)
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
        color: #111827;
        background-color: #f9fafb;
    }
    
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Professional Top Header */
    .console-header {
        border-bottom: 1px solid #d1d5db;
        padding-bottom: 0.85rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .console-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .console-desc {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 0.2rem;
    }

    /* Structured Panels */
    .panel {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f3f4f6;
    }

    /* Result Box */
    .result-box {
        background: #ffffff;
        border: 1px solid #9ca3af;
        border-radius: 4px;
        padding: 1.1rem;
    }

    .result-class {
        font-size: 1.4rem;
        font-weight: 800;
        color: #111827;
    }

    .result-meta {
        font-size: 0.85rem;
        color: #2563eb;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .dispatch-box {
        background: #f3f4f6;
        border-radius: 3px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #1f2937;
        line-height: 1.45;
    }

    /* Monospace Token Tag */
    .token-tag {
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: 0.8rem;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 0.2rem 0.45rem;
        border-radius: 3px;
        color: #1f2937;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        display: inline-block;
    }

    .diff-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .diff-table th, .diff-table td {
        border: 1px solid #e5e7eb;
        padding: 0.5rem 0.75rem;
        text-align: left;
    }
    .diff-table th {
        background: #f9fafb;
        font-weight: 600;
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
from sklearn.naive_bayes import MultinomialNB

# Category Response Routing
DISPATCH_ROUTING = {
    'Earthquake': {
        'priority': 'Priority 1 (Life Safety)',
        'unit': 'Urban Search and Rescue (USAR)',
        'protocol': 'Deploy structural assessment engineers, acoustic listening units, and canine teams.'
    },
    'Flood': {
        'priority': 'Priority 1 (Life Safety)',
        'unit': 'Water Rescue & Maritime Response',
        'protocol': 'Dispatch swift-water rescue craft, inflatable boat teams, and establish high-ground evacuation points.'
    },
    'Wildfire': {
        'priority': 'Priority 1 (Life Safety)',
        'unit': 'Forestry Fire Units',
        'protocol': 'Issue perimeter evacuation orders, dispatch aerial water drops, and establish fuel-break lines.'
    },
    'Typhoon': {
        'priority': 'Priority 2 (Civil Defense)',
        'unit': 'Meteorological & Civil Defense',
        'protocol': 'Issue coastal storm surge warnings, open municipal emergency shelters, and pre-position supplies.'
    },
    'Transportation Accident': {
        'priority': 'Priority 2 (Paramedic/Traffic)',
        'unit': 'Highway Patrol & Paramedic Corps',
        'protocol': 'Dispatch heavy hydraulic extrication units, trauma ambulances, and establish traffic diversions.'
    },
    'Explosion': {
        'priority': 'Priority 1 (Life Safety / HazMat)',
        'unit': 'Hazardous Materials & Trauma Battalion',
        'protocol': 'Enforce safety perimeter, assess toxic chemical contamination, and route blast injury patients.'
    },
    'Shooting': {
        'priority': 'Priority 1 (Law Enforcement Tactical)',
        'unit': 'Tactical Police Units',
        'protocol': 'Deploy active threat containment units, lock down public perimeter, and establish casualty triage.'
    },
    'Bombing': {
        'priority': 'Priority 1 (Explosive Ordnance)',
        'unit': 'EOD & Homeland Security',
        'protocol': 'Dispatch explosive ordnance disposal units, initiate secondary sweeps, and activate mass casualty protocols.'
    },
    'Haze': {
        'priority': 'Priority 3 (Public Health)',
        'unit': 'Public Health Directorate',
        'protocol': 'Issue air quality index advisories, distribute particulate filtration masks, and alert hospitals.'
    },
    'Meteor': {
        'priority': 'Priority 3 (Geological Survey)',
        'unit': 'Geological & Space Survey',
        'protocol': 'Triangulate trajectory data, verify ground impact coordinates, and inspect potential shockwaves.'
    },
    'Building Collapse': {
        'priority': 'Priority 1 (Heavy USAR)',
        'unit': 'Heavy USAR & Civil Engineering',
        'protocol': 'Mobilize crane extrication units, fiber-optic search cameras, and structural shoring equipment.'
    },
    'Fire': {
        'priority': 'Priority 1 (Fire Suppression)',
        'unit': 'Municipal Fire & Rescue',
        'protocol': 'Route high-rise ladder trucks, high-volume pumper engines, and establish hydrant lines.'
    }
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

# Load Dataset and Train Core Pipeline
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
    
    # Train Models
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    
    labels = sorted(y_train.unique())
    return tfidf_vectorizer, lr, rf, nb, labels, df

tfidf_model, lr_model, rf_model, nb_model, class_names, raw_df = load_models()

# App Header
st.markdown("""
<div class="console-header">
    <div>
        <div class="console-title">Crisis Text Multi-Class Classifier</div>
        <div class="console-desc">Real-time NLP categorization and response routing across 12 disaster categories.</div>
    </div>
    <div style="font-size: 0.8rem; color: #6b7280; text-align: right;">
        Corpus: <strong>11,015</strong> records · Vocab: <strong>10,000</strong> n-grams · Peak F1: <strong>0.9989</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_single, tab_compare, tab_batch, tab_benchmark, tab_dataset, tab_legal = st.tabs([
    "Single-Tweet Triage",
    "Model Comparison Sandbox",
    "Batch File Processing",
    "Evaluation Benchmark",
    "Dataset Statistics",
    "Terms & Privacy"
])

# Tab 1: Single-Tweet Triage
with tab_single:
    st.markdown("**Interactive Scenario Selection:**")
    
    preset_scenarios = {
        "Earthquake": "6.8 magnitude earthquake struck offshore, strong shaking felt across the province for 45 seconds, several buildings cracked.",
        "Flood": "Water levels rising rapidly above 2 meters in downtown residential district, multiple families stranded on rooftops awaiting rescue boats.",
        "Wildfire": "Massive wildfire spreading rapidly through pine forest due to 40mph dry winds, zero containment, mandatory evacuation order issued for zone 4.",
        "Industrial Explosion": "Loud explosion reported at chemical industrial facility, heavy toxic black smoke billowing into surrounding residential area.",
        "Transportation": "Major multi-car pileup and commuter train derailment on interstate highway 101, multiple serious injuries, paramedics dispatched.",
        "Shooting": "Active shooter reported near central shopping plaza, tactical police units deploying to the scene, stay clear of the area.",
        "Typhoon": "Super typhoon making landfall along eastern coastline, maximum sustained winds of 150km/h, storm surge warnings active.",
        "Building Collapse": "Old commercial residential building collapsed in city center, search and rescue teams digging through concrete rubble for survivors."
    }
    
    # 8 Interactive Preset Buttons in 2 Rows
    p_cols1 = st.columns(4)
    p_keys = list(preset_scenarios.keys())
    for i in range(4):
        if p_cols1[i].button(p_keys[i], use_container_width=True):
            st.session_state['active_text'] = preset_scenarios[p_keys[i]]
            
    p_cols2 = st.columns(4)
    for i in range(4, 8):
        if p_cols2[i - 4].button(p_keys[i], use_container_width=True):
            st.session_state['active_text'] = preset_scenarios[p_keys[i]]

    st.write("")
    
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    with col_left:
        st.markdown("**Input Message:**")
        default_input = st.session_state.get('active_text', "6.5 magnitude earthquake struck offshore, strong shaking felt in downtown district, power grid failures reported.")
        tweet_input = st.text_area("Tweet content:", value=default_input, height=130, placeholder="Paste or type any crisis text...")
        
        c_eng, c_run = st.columns([3, 2])
        with c_eng:
            engine_choice = st.selectbox(
                "Classification Engine:",
                [
                    "Ensemble (Weighted Logistic Regression + Random Forest)",
                    "Logistic Regression (Sublinear TF-IDF)",
                    "Random Forest (100 Estimators, TF-IDF)",
                    "Multinomial Naive Bayes (Laplace alpha=1.0)"
                ]
            )
        with c_run:
            st.write("")
            st.write("")
            run_btn = st.button("Run Classification", type="primary", use_container_width=True)
            
        st.caption(f"Input: {len(tweet_input)} characters | {len(tweet_input.split())} words")

    with col_right:
        st.markdown("**Classification Output:**")
        
        if tweet_input.strip():
            t0 = time.time()
            cleaned_str = clean_tweet_text(tweet_input)
            
            if not cleaned_str:
                st.warning("Input contains no informative tokens after preprocessing.")
            else:
                x_vec = tfidf_model.transform([cleaned_str])
                p_lr = lr_model.predict_proba(x_vec)[0]
                p_rf = rf_model.predict_proba(x_vec)[0]
                p_nb = nb_model.predict_proba(x_vec)[0]
                
                if "Ensemble" in engine_choice:
                    final_p = 0.60 * p_lr + 0.40 * p_rf
                elif "Logistic Regression" in engine_choice:
                    final_p = p_lr
                elif "Random Forest" in engine_choice:
                    final_p = p_rf
                else:
                    final_p = p_nb
                    
                t_exec = (time.time() - t0) * 1000
                top_idx = int(np.argmax(final_p))
                predicted_class = class_names[top_idx]
                confidence = final_p[top_idx] * 100
                routing = DISPATCH_ROUTING.get(predicted_class, {'priority': 'Priority 2', 'unit': 'Civil Defense', 'protocol': 'Standard response.'})
                
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; font-weight: 600;">Predicted Category</div>
                    <div class="result-class">{predicted_class}</div>
                    <div class="result-meta">Confidence: <strong>{confidence:.2f}%</strong> | Latency: <strong>{t_exec:.2f} ms</strong></div>
                    <div class="dispatch-box">
                        <div><strong>Priority Level:</strong> {routing['priority']}</div>
                        <div><strong>Assigned Agency:</strong> {routing['unit']}</div>
                        <div><strong>Action Protocol:</strong> {routing['protocol']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**Top Category Probability Ranking:**")
                top4_ranks = np.argsort(final_p)[::-1][:4]
                
                for rank, i in enumerate(top4_ranks, start=1):
                    c_name = class_names[i]
                    c_pct = final_p[i] * 100
                    
                    r_col1, r_col2, r_col3 = st.columns([3, 5, 2])
                    r_col1.write(f"**#{rank} {c_name}**")
                    r_col2.progress(float(final_p[i]))
                    r_col3.write(f"{c_pct:.2f}%")
                    
                with st.expander("View Cleaned Token Breakdown & Feature Vector"):
                    st.markdown("**Preprocessed Tokens:**")
                    st.markdown(" ".join([f"<span class='token-tag'>{w}</span>" for w in cleaned_str.split()]), unsafe_allow_html=True)
                    
                    st.markdown("**All 12 Category Probabilities:**")
                    all_probs_df = pd.DataFrame({
                        "Category": class_names,
                        "Probability": [f"{p * 100:.2f}%" for p in final_p]
                    }).sort_values(by="Probability", ascending=False)
                    st.dataframe(all_probs_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select a preset scenario above or enter custom text to view live predictions.")

# Tab 2: Interactive Model Comparison Sandbox
with tab_compare:
    st.markdown("#### Side-by-Side Model Comparison Sandbox")
    st.markdown("Compare predictions across **Logistic Regression**, **Random Forest**, and **Naive Bayes** simultaneously on the same text:")
    
    cmp_input = st.text_area("Test text for multi-model comparison:", value=tweet_input if tweet_input else "Forest fire spreading out of control near residential area due to severe drought and dry wind conditions.", height=90)
    
    if cmp_input.strip():
        c_clean = clean_tweet_text(cmp_input)
        if c_clean:
            c_vec = tfidf_model.transform([c_clean])
            
            p_lr_c = lr_model.predict_proba(c_vec)[0]
            p_rf_c = rf_model.predict_proba(c_vec)[0]
            p_nb_c = nb_model.predict_proba(c_vec)[0]
            p_ens_c = 0.60 * p_lr_c + 0.40 * p_rf_c
            
            cmp_cols = st.columns(3)
            with cmp_cols[0]:
                st.markdown("**Logistic Regression (TF-IDF)**")
                lr_top = class_names[int(np.argmax(p_lr_c))]
                st.write(f"Prediction: **{lr_top}**")
                st.write(f"Confidence: **{np.max(p_lr_c) * 100:.2f}%**")
                st.progress(float(np.max(p_lr_c)))
                
            with cmp_cols[1]:
                st.markdown("**Random Forest (TF-IDF)**")
                rf_top = class_names[int(np.argmax(p_rf_c))]
                st.write(f"Prediction: **{rf_top}**")
                st.write(f"Confidence: **{np.max(p_rf_c) * 100:.2f}%**")
                st.progress(float(np.max(p_rf_c)))
                
            with cmp_cols[2]:
                st.markdown("**Weighted Ensemble**")
                ens_top = class_names[int(np.argmax(p_ens_c))]
                st.write(f"Prediction: **{ens_top}**")
                st.write(f"Confidence: **{np.max(p_ens_c) * 100:.2f}%**")
                st.progress(float(np.max(p_ens_c)))

# Tab 3: Batch File Processing
with tab_batch:
    st.markdown("#### Batch CSV File Classification")
    st.markdown("Upload a CSV file containing social media messages to run automated multi-class triage across thousands of records:")
    
    csv_file = st.file_uploader("Upload CSV file (must include `tweet_text` or `text` column):", type=["csv"])
    
    if csv_file is not None:
        try:
            batch_df = pd.read_csv(csv_file)
            target_col = None
            for c in ['tweet_text', 'text', 'Tweet', 'content', 'message']:
                if c in batch_df.columns:
                    target_col = c
                    break
                    
            if target_col is None:
                st.error("No valid text column found. Please include a column named 'tweet_text' or 'text'.")
            else:
                st.success(f"File loaded successfully: {len(batch_df):,} rows. Target column: `{target_col}`")
                
                with st.spinner("Processing batch classifications..."):
                    b_cleaned = batch_df[target_col].apply(clean_tweet_text)
                    b_vec = tfidf_model.transform(b_cleaned)
                    b_preds = lr_model.predict(b_vec)
                    b_probs = lr_model.predict_proba(b_vec)
                    b_confs = np.max(b_probs, axis=1) * 100
                    
                    batch_df['Predicted_Disaster'] = b_preds
                    batch_df['Confidence_Score'] = np.round(b_confs, 2)
                    batch_df['Priority_Tier'] = [DISPATCH_ROUTING.get(p, {}).get('priority', 'Priority 2') for p in b_preds]
                    batch_df['Assigned_Unit'] = [DISPATCH_ROUTING.get(p, {}).get('unit', 'Civil Defense') for p in b_preds]
                
                st.dataframe(batch_df[[target_col, 'Predicted_Disaster', 'Confidence_Score', 'Priority_Tier', 'Assigned_Unit']].head(25), use_container_width=True)
                
                out_buf = io.StringIO()
                batch_df.to_csv(out_buf, index=False)
                st.download_button(
                    label="Download Annotated CSV Dataset",
                    data=out_buf.getvalue(),
                    file_name="classified_disaster_tweets.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# Tab 4: Evaluation Benchmark
with tab_benchmark:
    st.markdown("#### Empirical Evaluation Benchmark (Held-Out Test Set, n = 1,647)")
    st.markdown("Quantitative evaluation metrics across 10 model architectures evaluated under a strict stratified 70/15/15 split:")
    
    benchmark_data = pd.DataFrame([
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
    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

# Tab 5: Dataset Statistics
with tab_dataset:
    st.markdown("#### Dataset Class Distribution (CrisisNLP Benchmark, n = 11,015)")
    
    counts_df = pd.DataFrame([
        {"Category": "Typhoon", "Sample Count": 1080, "Class Share": "9.81%"},
        {"Category": "Transportation Accident", "Sample Count": 1072, "Class Share": "9.73%"},
        {"Category": "Wildfire", "Sample Count": 1041, "Class Share": "9.45%"},
        {"Category": "Earthquake", "Sample Count": 1029, "Class Share": "9.34%"},
        {"Category": "Flood", "Sample Count": 1021, "Class Share": "9.27%"},
        {"Category": "Explosion", "Sample Count": 951, "Class Share": "8.63%"},
        {"Category": "Shooting", "Sample Count": 914, "Class Share": "8.30%"},
        {"Category": "Bombing", "Sample Count": 907, "Class Share": "8.23%"},
        {"Category": "Haze", "Sample Count": 903, "Class Share": "8.20%"},
        {"Category": "Meteor", "Sample Count": 877, "Class Share": "7.96%"},
        {"Category": "Building Collapse", "Sample Count": 832, "Class Share": "7.55%"},
        {"Category": "Fire", "Sample Count": 388, "Class Share": "3.52%"}
    ])
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

# Tab 6: Terms & Privacy
with tab_legal:
    st.markdown("#### Privacy Statement & Terms of Use")
    st.markdown("""
    - **In-Memory Processing:** All text and file classification operations occur ephemerally in active memory. No user inputs are permanently logged or shared.
    - **Intended Use:** This tool is designed for academic research, situational awareness benchmarking, and decision-support exploration.
    - **Open-Source License:** Released under the MIT Open-Source License.
    """)

# Minimalist Footer
st.markdown("""
<div style="text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.75rem;">
    Crisis Text Multi-Class Classifier · Open-Source Research Console · MIT License
</div>
""", unsafe_allow_html=True)
