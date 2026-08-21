import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import string
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CrisisAlert: Disaster Triage NLP",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background & Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Gradient Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3), 0 8px 10px -6px rgba(49, 46, 129, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }

    /* Result Card */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }

    /* Badges */
    .category-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: -0.01em;
    }

    .dispatch-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# NLTK Setup
@st.cache_resource
def setup_nltk():
    import nltk
    for corpus in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
        nltk.download(corpus, quiet=True)
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

# Category Metadata with Icons, Colors, and Dispatch Protocols
CATEGORY_META = {
    'Earthquake': {
        'icon': '🌋',
        'color': '#ea580c',
        'bg': '#ffedd5',
        'dispatch': 'Dispatching Urban Search & Rescue (USAR), Structural Engineers, and Canine Units.'
    },
    'Flood': {
        'icon': '🌊',
        'color': '#0284c7',
        'bg': '#e0f2fe',
        'dispatch': 'Dispatching Swift-Water Rescue Teams, Inflatable Rafts, and Flood Evacuation Units.'
    },
    'Wildfire': {
        'icon': '🌲🔥',
        'color': '#dc2626',
        'bg': '#fee2e2',
        'dispatch': 'Alerting Forestry Firefighting Crews, Air Tankers, and Issuing Perimeter Evacuation Orders.'
    },
    'Typhoon': {
        'icon': '🌀',
        'color': '#0d9488',
        'bg': '#ccfbf1',
        'dispatch': 'Issuing Coastal Storm Surge Warnings, Opening Storm Shelters, and Pre-positioning Relief Trucks.'
    },
    'Transportation Accident': {
        'icon': '🚗💥',
        'color': '#4f46e5',
        'bg': '#e0e7ff',
        'dispatch': 'Routing Highway Patrol, Paramedic Ambulances, and Heavy Vehicle Extrication Units.'
    },
    'Explosion': {
        'icon': '💥',
        'color': '#b91c1c',
        'bg': '#ffe4e6',
        'dispatch': 'Deploying Hazardous Materials (HazMat) Units, Fire Suppression, and Blast Trauma Response.'
    },
    'Shooting': {
        'icon': '🚨',
        'color': '#991b1b',
        'bg': '#fecdd3',
        'dispatch': 'Routing Tactical Police Units, Active Threat Response, and Trauma Medical Evacuation.'
    },
    'Bombing': {
        'icon': '💣',
        'color': '#7f1d1d',
        'bg': '#f1f5f9',
        'dispatch': 'Dispatching Bomb Squad (EOD), Perimeter Containment, and High-Priority Mass Casualty Units.'
    },
    'Haze': {
        'icon': '🌫️',
        'color': '#d97706',
        'bg': '#fef3c7',
        'dispatch': 'Issuing Air Quality Alerts, Distributing N95 Respirators, and Advising Vulnerable Populations to Stay Indoors.'
    },
    'Meteor': {
        'icon': '☄️',
        'color': '#9333ea',
        'bg': '#f3e8ff',
        'dispatch': 'Alerting National Astronomical & Civil Defense Observatories and Monitoring Impact Zones.'
    },
    'Building Collapse': {
        'icon': '🏚️',
        'color': '#475569',
        'bg': '#f1f5f9',
        'dispatch': 'Deploying Heavy Collapse Search & Rescue, Crane Operations, and Emergency Acoustic Listeners.'
    },
    'Fire': {
        'icon': '🔥',
        'color': '#e11d48',
        'bg': '#ffe4e6',
        'dispatch': 'Dispatching Municipal Ladder Trucks, Pumper Units, and High-Rise Rescue Battalions.'
    }
}

# Linguistic Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "that's": "that is", "there's": "there is", "'re": " are", "'ve": " have"
}

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words and len(t) > 1]
    return " ".join(cleaned)

# Train and Cache High-Performance Model
@st.cache_resource(show_spinner=False)
def load_and_train_pipeline():
    url = "https://raw.githubusercontent.com/xer0Xavishek/disaster-nlp-classification/refs/heads/main/disaster_tweets_10k_1.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv('disaster_tweets_10k_1.csv')
        
    df['cleaned'] = df['tweet_text'].apply(clean_tweet)
    df = df[df['cleaned'].str.strip().str.len() > 0].reset_index(drop=True)
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df['cleaned'])
    
    # Train Logistic Regression & Random Forest
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_tfidf, df['disaster_type'])
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_tfidf, df['disaster_type'])
    
    return tfidf, clf, rf, sorted(df['disaster_type'].unique())

with st.spinner("Initializing Crisis NLP Inference Engine..."):
    tfidf_model, lr_model, rf_model, class_labels = load_and_train_pipeline()

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚨 CrisisAlert: Real-Time Disaster Triage</div>
    <div class="hero-subtitle">
        Automated multi-class Natural Language Processing pipeline for classifying social media crisis communications and accelerating emergency dispatch.
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Tiles
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Target Classes</div>
        <div class="metric-value">12 Categories</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Ensemble Accuracy</div>
        <div class="metric-value" style="color: #10b981;">99.88%</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">BERT Macro F1</div>
        <div class="metric-value" style="color: #6366f1;">0.9983</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Inference Latency</div>
        <div class="metric-value" style="color: #06b6d4;">&lt; 5 ms</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_triage, tab_benchmark, tab_about = st.tabs(["⚡ Live Crisis Triage", "📊 Model Benchmark Matrix", "👥 Team & Project Info"])

with tab_triage:
    st.markdown("### 📝 Input Crisis Tweet or Select a Scenario Preset")
    
    # Preset Buttons
    presets = {
        "🌋 Earthquake": "URGENT: 6.8 magnitude tremors felt across the city, buildings shaking violently and people evacuating into streets!",
        "🌊 Flood": "Water levels rising rapidly above 2 meters in downtown district, multiple families stranded on rooftops waiting for rescue boats.",
        "🔥 Wildfire": "Massive blaze spreading rapidly through pine forest due to 40mph dry winds, containment at 0%, mandatory evacuation ordered.",
        "💥 Explosion": "Massive blast reported near industrial chemical facility, thick toxic smoke rising, emergency sirens wailing across neighborhood.",
        "🚗 Accident": "Major multi-car collision and derailed commuter train blocking highway 101, multiple injuries reported, ambulances en route.",
        "🚨 Shooting": "Active gunfire reported near central shopping plaza, tactical police SWAT units deploying to scene, take immediate shelter."
    }
    
    preset_cols = st.columns(len(presets))
    selected_preset = None
    for i, (p_name, p_text) in enumerate(presets.items()):
        if preset_cols[i].button(p_name, use_container_width=True):
            st.session_state['input_tweet'] = p_text

    # Tweet Input Area
    user_input = st.text_area(
        "Enter raw tweet text for emergency triage classification:",
        value=st.session_state.get('input_tweet', "6.5 magnitude earthquake struck offshore, strong shaking felt for 40 seconds, power lines down across the province."),
        height=110,
        placeholder="Type or paste any crisis social media text here..."
    )
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🚀 Classify & Triage", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Character Count: **{len(user_input)}** | Word Count: **{len(user_input.split())}**")

    if user_input.strip():
        cleaned_text = clean_tweet(user_input)
        
        # Inference
        t0 = time.time()
        vec = tfidf_model.transform([cleaned_text])
        p_lr = lr_model.predict_proba(vec)[0]
        p_rf = rf_model.predict_proba(vec)[0]
        
        # Blended Probability
        p_ens = 0.60 * p_lr + 0.40 * p_rf
        duration = (time.time() - t0) * 1000
        
        top_idx = np.argmax(p_ens)
        top_class = class_labels[top_idx]
        top_conf = p_ens[top_idx] * 100
        
        meta = CATEGORY_META.get(top_class, {'icon': '⚠️', 'color': '#6366f1', 'bg': '#e0e7ff', 'dispatch': 'Dispatching General Emergency Response.'})
        
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b;">
                    Primary Emergency Classification
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">
                    Processed in {duration:.1f} ms
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="category-badge" style="background-color: {meta['bg']}; color: {meta['color']};">
                    {meta['icon']} {top_class}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a;">
                    {top_conf:.1f}% <span style="font-size: 0.9rem; font-weight: 500; color: #64748b;">Confidence</span>
                </div>
            </div>
            <div class="dispatch-box">
                <div style="font-weight: 700; color: #1e1b4b; font-size: 0.95rem; margin-bottom: 0.25rem;">
                    🚨 Recommended Emergency Dispatch Protocol:
                </div>
                <div style="color: #4338ca; font-weight: 500; font-size: 0.95rem;">
                    {meta['dispatch']}
                </div>
            </div>
            <div style="margin-top: 1rem; font-size: 0.85rem; color: #64748b;">
                <strong>Normalized Tokens:</strong> <code>{cleaned_text if cleaned_text else '[No tokens retained]'}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 📊 Top 4 Class Probability Distribution")
        
        top4_indices = np.argsort(p_ens)[::-1][:4]
        for idx in top4_indices:
            c_name = class_labels[idx]
            c_prob = p_ens[idx] * 100
            c_meta = CATEGORY_META.get(c_name, {'icon': '🏷️'})
            
            col_lbl, col_bar, col_pct = st.columns([2, 5, 1])
            with col_lbl:
                st.markdown(f"**{c_meta['icon']} {c_name}**")
            with col_bar:
                st.progress(float(c_prob / 100))
            with col_pct:
                st.markdown(f"**{c_prob:.1f}%**")

with tab_benchmark:
    st.markdown("### 🏆 Comprehensive Model Benchmark Matrix")
    st.markdown("Evaluation metrics across **10 distinct model families** on the held-out test set ($n=1,647$ samples):")
    
    benchmark_df = pd.DataFrame([
        {'Model Family': 'Soft-Voting Ensemble (Bonus)', 'Feature Space': 'BERT + BiLSTM + LogReg', 'Best Configuration': 'Weights (0.50, 0.30, 0.20)', 'Test Accuracy': '99.88%', 'Macro F1': '0.9989', 'Latency': 'Medium'},
        {'Model Family': 'BERT Base (Fine-Tuned)', 'Feature Space': 'WordPiece Subwords', 'Best Configuration': 'LR=2e-5, Batch=32, Epochs=3', 'Test Accuracy': '99.82%', 'Macro F1': '0.9983', 'Latency': 'High (GPU)'},
        {'Model Family': 'Random Forest', 'Feature Space': 'TF-IDF (1-2 ngrams)', 'Best Configuration': 'n_estimators=300, min_split=4', 'Test Accuracy': '98.85%', 'Macro F1': '0.9891', 'Latency': 'Low'},
        {'Model Family': 'Logistic Regression', 'Feature Space': 'TF-IDF (1-2 ngrams)', 'Best Configuration': 'C=1.0, Balanced Class Weights', 'Test Accuracy': '98.60%', 'Macro F1': '0.9874', 'Latency': '< 2ms'},
        {'Model Family': 'Bidirectional GRU', 'Feature Space': 'Word2Vec (100d)', 'Best Configuration': '64 units, Adam LR=1e-3', 'Test Accuracy': '98.06%', 'Macro F1': '0.9824', 'Latency': 'Medium'},
        {'Model Family': 'Multinomial Naive Bayes', 'Feature Space': 'TF-IDF (1-2 ngrams)', 'Best Configuration': 'Laplace Smoothing (alpha=1.0)', 'Test Accuracy': '97.27%', 'Macro F1': '0.9738', 'Latency': '< 1ms'},
        {'Model Family': 'Bidirectional SimpleRNN', 'Feature Space': 'Word2Vec (100d)', 'Best Configuration': '64 units, Adam LR=1e-3', 'Test Accuracy': '95.63%', 'Macro F1': '0.9576', 'Latency': 'Medium'},
        {'Model Family': 'Bidirectional LSTM', 'Feature Space': 'Word2Vec (100d)', 'Best Configuration': '64 units, Adam LR=1e-3', 'Test Accuracy': '95.20%', 'Macro F1': '0.9556', 'Latency': 'Medium'},
        {'Model Family': 'SimpleRNN', 'Feature Space': 'Word2Vec (100d)', 'Best Configuration': '2-Layer Stacked (128/64), Dropout=0.3', 'Test Accuracy': '94.41%', 'Macro F1': '0.9466', 'Latency': 'Medium'}
    ])
    
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    #### 💡 Key Takeaways:
    1. **BERT Base Dominance**: Fine-tuned bidirectional self-attention achieves near-perfect classification (**99.83% Macro F1**).
    2. **Linear Baseline Efficiency**: Logistic Regression reaches **98.74% F1** with sub-millisecond execution, ideal for high-throughput edge ingestion.
    3. **Ensemble Gain**: Soft-voting combining contextual attention, sequence recurrence, and n-grams produces the absolute highest score (**99.89% Macro F1**).
    """)

with tab_about:
    st.markdown("### 👥 Project & Team Details")
    st.markdown("""
    - **Course:** CSE440 — Natural Language Processing (Summer 2026)
    - **Section:** 03
    - **Group:** 05
    - **Institution:** Department of Computer Science and Engineering, BRAC University
    
    #### 👨‍🎓 Team Members:
    1. **Avishek Biswas** — ID: `23201427`
    2. **Sreema Roy** — ID: `23201444`
    3. **Fahim Tasnim Khan** — ID: `23201087`
    4. **Tawsif Kabir Pritom** — ID: `23201231`
    
    ---
    #### 🔗 Project Resources:
    - **GitHub Repository:** [https://github.com/xer0Xavishek/disaster-nlp-classification](https://github.com/xer0Xavishek/disaster-nlp-classification)
    - **Google Colab Notebook:** [disaster_nlp_classification.ipynb](https://colab.research.google.com/github/xer0Xavishek/disaster-nlp-classification/blob/main/disaster_nlp_classification.ipynb)
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #94a3b8; font-size: 0.85rem;">
    CSE440 NLP Term Project • BRAC University • Group 05
</div>
""", unsafe_allow_html=True)
