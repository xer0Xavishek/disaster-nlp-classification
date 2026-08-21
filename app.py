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
    page_title="CrisisNLP: Real-Time Disaster Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Open-Source CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1e293b;
        background-color: #f8fafc;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }

    /* Top Navigation Header */
    .top-header {
        background: #0f172a;
        color: #ffffff;
        padding: 1.25rem 1.75rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .top-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .top-header-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* Status & KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.2rem;
    }

    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Classification Result Box */
    .result-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 1.25rem;
        margin-top: 0.5rem;
    }

    .result-badge {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }

    .result-sub {
        font-size: 0.9rem;
        color: #2563eb;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .dispatch-container {
        background: #f1f5f9;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        font-size: 0.88rem;
    }

    .code-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: #e2e8f0;
        padding: 0.2rem 0.45rem;
        border-radius: 4px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK Packages
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

# Category Mapping, Priority Levels, and Emergency Dispatch Protocols
CATEGORY_DETAILS = {
    'Earthquake': {
        'priority': 'Priority 1 (Immediate Life Safety)',
        'agency': 'Urban Search & Rescue (USAR)',
        'action': 'Deploy structural assessment teams, acoustic listening units, and canine search teams to damaged zones.'
    },
    'Flood': {
        'priority': 'Priority 1 (Immediate Life Safety)',
        'agency': 'Water Rescue & Maritime Response',
        'action': 'Dispatch swift-water rescue craft, inflatable boat teams, and establish high-ground evacuation points.'
    },
    'Wildfire': {
        'priority': 'Priority 1 (Immediate Life Safety)',
        'agency': 'Forestry & Regional Fire Units',
        'action': 'Issue perimeter evacuation zones, dispatch aerial water drops, and establish fuel-break defense lines.'
    },
    'Typhoon': {
        'priority': 'Priority 2 (Civil Defense & Infrastructure)',
        'agency': 'Meteorological & Civil Defense',
        'action': 'Issue coastal storm surge warnings, open municipal emergency shelters, and pre-position emergency relief stocks.'
    },
    'Transportation Accident': {
        'priority': 'Priority 2 (Paramedic & Traffic Management)',
        'agency': 'Highway Patrol & Paramedic Corps',
        'action': 'Dispatch heavy hydraulic extrication units, trauma ambulances, and establish traffic diversions.'
    },
    'Explosion': {
        'priority': 'Priority 1 (Immediate Life Safety / HazMat)',
        'agency': 'Hazardous Materials & Trauma Battalion',
        'action': 'Enforce safety cordon, assess chemical contamination risks, and route burn and blast trauma patients.'
    },
    'Shooting': {
        'priority': 'Priority 1 (Law Enforcement Tactical)',
        'agency': 'Tactical Police & Emergency Medical',
        'action': 'Deploy active threat containment units, lock down public perimeter, and establish casualty collection points.'
    },
    'Bombing': {
        'priority': 'Priority 1 (Explosive Ordnance / Mass Casualty)',
        'agency': 'EOD & Homeland Security Units',
        'action': 'Dispatch explosive ordnance disposal units, initiate secondary device sweeps, and activate mass casualty protocols.'
    },
    'Haze': {
        'priority': 'Priority 3 (Public Health Advisory)',
        'agency': 'Environmental & Public Health Directorate',
        'action': 'Issue particulate air quality advisories (AQI), distribute N95 filtration masks, and protect vulnerable demographics.'
    },
    'Meteor': {
        'priority': 'Priority 3 (Scientific & Geological Assessment)',
        'agency': 'National Space & Geological Survey',
        'action': 'Triangulate trajectory data, verify ground impact coordinates, and inspect potential seismic shockwaves.'
    },
    'Building Collapse': {
        'priority': 'Priority 1 (Heavy Search & Rescue)',
        'agency': 'Heavy USAR & Civil Engineering',
        'action': 'Mobilize crane extrication units, fiber-optic search cameras, and structural shoring equipment.'
    },
    'Fire': {
        'priority': 'Priority 1 (Fire Suppression)',
        'agency': 'Municipal Fire & Rescue Service',
        'action': 'Route high-rise ladder trucks, high-output pumper units, and establish hydrant supply lines.'
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
def load_classification_pipeline():
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
    
    # Train Logistic Regression (Baseline Edge Model)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    labels = sorted(y_train.unique())
    return tfidf_vectorizer, lr, rf, labels, df

tfidf_model, lr_classifier, rf_classifier, class_names, raw_dataset = load_classification_pipeline()

# Application Header (Clean Open-Source Branding)
st.markdown("""
<div class="top-header">
    <div>
        <div class="top-header-title">CrisisNLP: Disaster Intelligence & Triage System</div>
        <div class="top-header-meta">Automated Multi-Class Social Media Crisis Categorization & First-Responder Dispatch Routing</div>
    </div>
    <div style="text-align: right;">
        <span style="background: #1e293b; padding: 0.35rem 0.75rem; border-radius: 4px; font-size: 0.8rem; border: 1px solid #334155;">
            Production Build · v1.0.0
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Benchmark Corpus</div>
        <div class="kpi-value">11,015 Tweets</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Incident Categories</div>
        <div class="kpi-value">12 Classes</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Peak Ensemble F1</div>
        <div class="kpi-value" style="color: #059669;">0.9989</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">BERT Base F1</div>
        <div class="kpi-value" style="color: #2563eb;">0.9983</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Tabs
nav_triage, nav_batch, nav_benchmark, nav_methodology, nav_about = st.tabs([
    "Single-Tweet Triage",
    "Batch File Processing",
    "Model Benchmark Matrix",
    "Methodology & Preprocessing",
    "About CrisisNLP"
])

# Tab 1: Single-Tweet Triage
with nav_triage:
    col_input, col_output = st.columns([11, 10], gap="large")
    
    with col_input:
        st.markdown("#### Input Text & Configuration")
        
        sample_dataset = {
            "Select a pre-loaded emergency scenario...": "",
            "Earthquake: Offshore 6.8 magnitude tremor with structural shaking": "URGENT: 6.8 magnitude earthquake struck offshore, strong shaking felt across the province for 45 seconds, several buildings cracked.",
            "Flood: Water levels rising above 2 meters with stranded residents": "Water levels rising rapidly above 2 meters in downtown residential district, multiple families stranded on rooftops awaiting rescue boats.",
            "Wildfire: Uncontained blaze driven by dry winds with evacuation order": "Massive wildfire spreading rapidly through pine forest due to 40mph dry winds, zero containment, mandatory evacuation order issued for zone 4.",
            "Industrial Explosion: Chemical plant detonation with smoke plume": "Loud explosion reported at chemical industrial facility, heavy toxic black smoke billowing into surrounding residential area.",
            "Transportation: Multi-vehicle pileup and train derailment": "Major multi-car pileup and commuter train derailment on interstate highway 101, multiple serious injuries, paramedics dispatched.",
            "Shooting: Active gunfire in commercial marketplace": "Active shooter reported near central shopping plaza, tactical police units deploying to the scene, stay clear of the area."
        }
        
        selected_scenario = st.selectbox("Scenario Presets:", list(sample_dataset.keys()))
        initial_text = sample_dataset[selected_scenario] if selected_scenario != "Select a pre-loaded emergency scenario..." else "6.5 magnitude earthquake struck offshore, strong shaking felt across city center, power grid failure reported."
        
        tweet_text_input = st.text_area(
            "Raw Tweet Content:",
            value=initial_text,
            height=130,
            placeholder="Paste or enter disaster-related social media text..."
        )
        
        c_opt1, c_opt2 = st.columns([3, 2])
        with c_opt1:
            chosen_engine = st.selectbox(
                "Classification Engine:",
                [
                    "Ensemble (Weighted Logistic Regression + Random Forest)",
                    "Logistic Regression (Sublinear TF-IDF)",
                    "Random Forest (TF-IDF, 100 Estimators)"
                ]
            )
        with c_opt2:
            st.write("")
            st.write("")
            classify_btn = st.button("Run Classification", type="primary", use_container_width=True)
            
        st.caption(f"Input Length: {len(tweet_text_input)} characters | Word Count: {len(tweet_text_input.split())} words")

    with col_output:
        st.markdown("#### Classification & Triage Output")
        
        if tweet_text_input.strip():
            start_time = time.time()
            cleaned_tokens_str = clean_tweet_text(tweet_text_input)
            
            if not cleaned_tokens_str:
                st.warning("No informative tokens remained after text normalization and stopword removal.")
            else:
                vectorized = tfidf_model.transform([cleaned_tokens_str])
                prob_lr = lr_classifier.predict_proba(vectorized)[0]
                prob_rf = rf_classifier.predict_proba(vectorized)[0]
                
                if "Ensemble" in chosen_engine:
                    final_probs = 0.60 * prob_lr + 0.40 * prob_rf
                elif "Logistic Regression" in chosen_engine:
                    final_probs = prob_lr
                else:
                    final_probs = prob_rf
                
                latency_ms = (time.time() - start_time) * 1000
                top_idx = int(np.argmax(final_probs))
                predicted_label = class_names[top_idx]
                confidence = final_probs[top_idx] * 100
                details = CATEGORY_DETAILS.get(predicted_label, {'priority': 'Priority 2', 'agency': 'Civil Defense', 'action': 'Standard assessment.'})
                
                st.markdown(f"""
                <div class="result-container">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">
                                Primary Predicted Category
                            </div>
                            <div class="result-badge">{predicted_label}</div>
                            <div class="result-sub">{confidence:.2f}% Confidence Score</div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; color: #64748b;">
                            Latency: <strong>{latency_ms:.2f} ms</strong>
                        </div>
                    </div>
                    <div class="dispatch-container">
                        <div style="font-weight: 600; color: #0f172a; margin-bottom: 0.25rem;">
                            Emergency Response Routing:
                        </div>
                        <div style="margin-bottom: 0.2rem;">
                            <strong>Priority:</strong> <span style="color: #dc2626; font-weight: 600;">{details['priority']}</span>
                        </div>
                        <div style="margin-bottom: 0.2rem;">
                            <strong>Lead Agency:</strong> {details['agency']}
                        </div>
                        <div>
                            <strong>Protocol:</strong> {details['action']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**Top Category Probability Distribution:**")
                top4_idx = np.argsort(final_probs)[::-1][:4]
                
                prob_data = []
                for rank, i in enumerate(top4_idx, start=1):
                    c_name = class_names[i]
                    c_pct = final_probs[i] * 100
                    prob_data.append({"Rank": f"#{rank}", "Category": c_name, "Probability": f"{c_pct:.2f}%", "Weight": final_probs[i]})
                    
                for item in prob_data:
                    c_col1, c_col2, c_col3 = st.columns([3, 5, 2])
                    with c_col1:
                        st.write(f"**{item['Category']}**")
                    with c_col2:
                        st.progress(float(item['Weight']))
                    with c_col3:
                        st.write(item['Probability'])
                
                st.markdown("**Processed Token Vector:**")
                token_list = cleaned_tokens_str.split()
                st.markdown(" ".join([f"<span class='code-pill'>{t}</span>" for t in token_list]), unsafe_allow_html=True)
        else:
            st.info("Provide tweet text on the left panel to generate real-time classification.")

# Tab 2: Batch File Processing
with nav_batch:
    st.markdown("#### Batch Tweet Processing")
    st.markdown("Upload a CSV file containing raw tweet text to classify multiple messages simultaneously.")
    
    uploaded_file = st.file_uploader("Upload CSV File (must contain a column named `tweet_text` or `text`):", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            target_col = None
            for c in ['tweet_text', 'text', 'Tweet', 'content']:
                if c in batch_df.columns:
                    target_col = c
                    break
            
            if target_col is None:
                st.error("Uploaded CSV must contain one of the following columns: `tweet_text`, `text`, `Tweet`, or `content`.")
            else:
                st.success(f"Loaded {len(batch_df)} rows. Processing text column: `{target_col}`")
                
                with st.spinner("Processing batch predictions..."):
                    batch_cleaned = batch_df[target_col].apply(clean_tweet_text)
                    batch_vec = tfidf_model.transform(batch_cleaned)
                    batch_preds = lr_classifier.predict(batch_vec)
                    batch_probs = lr_classifier.predict_proba(batch_vec)
                    batch_confs = np.max(batch_probs, axis=1) * 100
                    
                    batch_df['Predicted_Disaster_Type'] = batch_preds
                    batch_df['Confidence_Percent'] = np.round(batch_confs, 2)
                    batch_df['Priority_Level'] = [CATEGORY_DETAILS.get(p, {}).get('priority', 'Priority 2') for p in batch_preds]
                
                st.dataframe(batch_df[[target_col, 'Predicted_Disaster_Type', 'Confidence_Percent', 'Priority_Level']].head(20), use_container_width=True)
                
                # Download CSV
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Annotated CSV Results",
                    data=csv_buffer.getvalue(),
                    file_name="crisisnlp_batch_classified.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Tab 3: Model Evaluation Benchmark
with nav_benchmark:
    st.markdown("#### Test Set Performance Summary (Held-Out Test Set, n = 1,647)")
    st.markdown("Quantitative evaluation metrics across 10 model families under a stratified 70/15/15 split:")
    
    benchmark_data = pd.DataFrame([
        {"Model Family": "Soft-Voting Ensemble", "Feature Representation": "BERT + BiLSTM + LogReg", "Best Configuration": "Weights (0.50, 0.30, 0.20)", "Test Accuracy": "99.88%", "Macro Precision": "0.9988", "Macro Recall": "0.9989", "Macro F1": "0.9989"},
        {"Model Family": "BERT Base (Fine-Tuned)", "Feature Representation": "WordPiece Subwords", "Best Configuration": "Config 1 (LR=2e-5, Batch=32, Epochs=3)", "Test Accuracy": "99.82%", "Macro Precision": "0.9983", "Macro Recall": "0.9983", "Macro F1": "0.9983"},
        {"Model Family": "Random Forest", "Feature Representation": "TF-IDF (1-2 ngrams)", "Best Configuration": "Config 3 (n=300, min_split=4)", "Test Accuracy": "98.85%", "Macro Precision": "0.9887", "Macro Recall": "0.9895", "Macro F1": "0.9891"},
        {"Model Family": "Logistic Regression", "Feature Representation": "TF-IDF (1-2 ngrams)", "Best Configuration": "Config 2 (C=1.0, Balanced)", "Test Accuracy": "98.60%", "Macro Precision": "0.9876", "Macro Recall": "0.9874", "Macro F1": "0.9874"},
        {"Model Family": "Bidirectional GRU", "Feature Representation": "Word2Vec (100d)", "Best Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "98.06%", "Macro Precision": "0.9830", "Macro Recall": "0.9824", "Macro F1": "0.9824"},
        {"Model Family": "Multinomial Naive Bayes", "Feature Representation": "TF-IDF (1-2 ngrams)", "Best Configuration": "Config 3 (alpha=1.0)", "Test Accuracy": "97.27%", "Macro Precision": "0.9754", "Macro Recall": "0.9727", "Macro F1": "0.9738"},
        {"Model Family": "Bidirectional SimpleRNN", "Feature Representation": "Word2Vec (100d)", "Best Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "95.63%", "Macro Precision": "0.9592", "Macro Recall": "0.9576", "Macro F1": "0.9576"},
        {"Model Family": "Bidirectional LSTM", "Feature Representation": "Word2Vec (100d)", "Best Configuration": "Config 1 (64 units, Adam)", "Test Accuracy": "95.20%", "Macro Precision": "0.9547", "Macro Recall": "0.9556", "Macro F1": "0.9556"},
        {"Model Family": "SimpleRNN", "Feature Representation": "Word2Vec (100d)", "Best Configuration": "Config 3 (2-Layer Stacked)", "Test Accuracy": "94.41%", "Macro Precision": "0.9482", "Macro Recall": "0.9466", "Macro F1": "0.9466"}
    ])
    
    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Key Findings:**
    1. **Transformer Contextual Superiority:** Fine-tuned BERT Base achieved a near-perfect standalone Macro F1 of **0.9983**, misclassifying only 3 out of 1,647 test instances.
    2. **Multi-Model Soft Voting:** Blending BERT self-attention probabilities with BiLSTM sequential modeling and Logistic Regression n-gram representations achieved peak performance of **0.9989 Macro F1**.
    3. **Edge Deployment Feasibility:** Classical Logistic Regression with sublinear TF-IDF delivers **0.9874 Macro F1** with sub-2ms inference latency, making it ideal for low-power edge ingestion.
    """)

# Tab 4: Methodology & Preprocessing
with nav_methodology:
    st.markdown("#### NLP Preprocessing & Feature Engineering Pipeline")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        **1. Text Cleaning & Normalization Steps:**
        - **HTML Entity Unescaping:** Strips web artifacts (`&amp;` $\\rightarrow$ `&`).
        - **Regex URL & Handle Removal:** Strips `https?://\\S+` and `@username` to prevent overfitting on specific account IDs or web domains.
        - **Contraction Expansion:** Normalizes auxiliary verbs (`can't` $\\rightarrow$ `cannot`, `it's` $\\rightarrow$ `it is`).
        - **Hashtag De-symbolization:** Strips `#` while preserving semantic tokens (`#earthquake` $\\rightarrow$ `earthquake`).
        - **Case Folding & Punctuation Stripping:** Standardizes vocabulary casing and removes punctuation/digits.
        - **NLTK WordNet Lemmatization:** Morphological root reduction (`flooding` $\\rightarrow$ `flood`, `collapsed` $\\rightarrow$ `collapse`).
        """)
    with col_m2:
        st.markdown("""
        **2. Vectorization & Feature Engineering:**
        - **Sublinear TF-IDF:** Replaces raw term frequency with $1 + \\ln(\\text{TF})$ and extracts unigrams and bigrams ($n \\in \\{1, 2\\}$) up to 10,000 max features.
        - **Word2Vec Continuous Embeddings:** Domain-trained Skip-Gram architecture ($d=100$, window=5, min_count=2) capturing semantic vector proximity.
        - **WordPiece Tokenization:** Transformer subword decomposition handling Out-of-Vocabulary (OOV) tokens without information loss.
        """)

# Tab 5: About CrisisNLP (Clean Open-Source Project Profile)
with nav_about:
    st.markdown("#### About CrisisNLP")
    st.markdown("""
    **CrisisNLP** is an open-source NLP system designed to classify social media communications during emergencies and route prioritized response directives to humanitarian agencies and emergency services in real time.
    
    **Corpus Source:**
    - Benchmark dataset derived from CrisisNLP and CrisisBench disaster repositories.
    - 11,015 human-annotated microblog records across 12 distinct incident categories.
    
    **Repository & Master Notebook:**
    - **GitHub Repository:** [https://github.com/xer0Xavishek/disaster-nlp-classification](https://github.com/xer0Xavishek/disaster-nlp-classification)
    - **Master Google Colab Notebook:** [disaster_nlp_classification.ipynb](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb)
    - **License:** MIT Open-Source License
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;">
    CrisisNLP: Disaster Intelligence & Triage System · Open Source Project
</div>
""", unsafe_allow_html=True)
