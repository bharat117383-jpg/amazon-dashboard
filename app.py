import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PrimeView Analytics | Amazon Prime Video Insights",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Classic Midnight Theme (#080F1F, #101B32, #172943, #38BDF8, #8B7CF6, #E5C07B)
MIDNIGHT_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #080F1F;
        color: #F1F5F9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #101B32 !important;
        border-right: 1px solid #172943;
    }
    
    /* Main Header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #38BDF8 0%, #8B7CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .sub-title {
        color: #A8B8CC;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background-color: #172943;
        border: 1px solid #233A5E;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38BDF8;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #A8B8CC;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    /* Insight Cards & Dividers */
    .insight-card {
        background-color: #101B32;
        border-left: 4px solid #E5C07B;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        color: #F1F5F9;
    }
    
    hr {
        border-color: #172943 !important;
        margin: 2rem 0 !important;
    }

    /* Streamlit Widget Styling Overrides */
    div[data-baseweb="select"] > div {
        background-color: #172943 !important;
        border-color: #233A5E !important;
        color: #F1F5F9 !important;
    }
    div[data-testid="stForm"] {
        border-color: #172943 !important;
        background-color: #101B32 !important;
    }
    
    /* Plotly Chart Container Cards */
    .plot-container {
        background-color: #172943;
        border-radius: 12px;
        border: 1px solid #233A5E;
        padding: 1rem;
    }
</style>
"""
st.markdown(MIDNIGHT_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    possible_paths = [
        os.path.join("data", "titles.csv"),
        "titles.csv",
        os.path.join("..", "data", "titles.csv")
    ]
    
    df_raw = None
    for path in possible_paths:
        if os.path.exists(path):
            df_raw = pd.read_csv(path)
            break
            
    if df_raw is None:
        return None, None

    df = df_raw.copy()

    # Standardization of column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standardize expected columns
    col_mapping = {
        'title': 'title',
        'type': 'type',
        'release_year': 'release_year',
        'imdb_score': 'imdb_score',
        'imdb_votes': 'imdb_votes',
        'genres': 'genres',
        'production_countries': 'production_countries',
        'runtime': 'runtime',
        'age_certification': 'age_certification'
    }
    
    # Fill missing essential text fields
    if 'type' in df.columns:
        df['type'] = df['type'].astype(str).str.upper().str.strip()
        df['type'] = df['type'].replace({'MOVIE': 'MOVIE', 'SHOW': 'TV SHOW', 'TV SHOW': 'TV SHOW'})
    else:
        df['type'] = 'UNKNOWN'

    if 'title' in df.columns:
        df['title'] = df['title'].fillna('Unknown Title')
    else:
        df['title'] = 'Unknown Title'

    # Safe parsing of stringified lists (e.g. "['documentation', 'comedy']")
    def safe_eval_list(val):
        if pd.isna(val) or not val:
            return []
        if isinstance(val, list):
            return val
        val_str = str(val).strip()
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                parsed = ast.literal_eval(val_str)
                return [str(x).strip() for x in parsed if x]
            except Exception:
                pass
        # Fallback split for comma separated values
        cleaned = val_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
        return [x.strip() for x in cleaned.split(',') if x.strip()]

    if 'genres' in df.columns:
        df['parsed_genres'] = df['genres'].apply(safe_eval_list)
    else:
        df['parsed_genres'] = [[] for _ in range(len(df))]

    if 'production_countries' in df.columns:
        df['parsed_countries'] = df['production_countries'].apply(safe_eval_list)
    else:
        df['parsed_countries'] = [[] for _ in range(len(df))]

    # Numeric Coercion
    for num_col in ['release_year', 'imdb_score', 'imdb_votes', 'runtime']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    if 'age_certification' in df.columns:
        df['age_certification'] = df['age_certification'].fillna('Unrated')
    else:
        df['age_certification'] = 'Unrated'

    # Optional Join with credits.csv if present
    credits_path = os.path.join("data", "credits.csv")
    credits_df = None
    if os.path.exists(credits_path):
        try:
            credits_df = pd.read_csv(credits_path)
            credits_df.columns = [c.strip().lower() for c in credits_df.columns]
        except Exception:
            credits_df = None

    return df, credits_df

df_raw, df_credits = load_and_preprocess_data()

if df_raw is None:
    st.error("🚨 Dataset `titles.csv` not found! Please make sure `titles.csv` is placed inside the `data/` folder.")
    st.stop()

# Plotly Theme Setup
PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor='#172943',
    plot_bgcolor='#172943',
    font=dict(color='#F1F5F9', family='Inter, sans-serif'),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(font=dict(color='#A8B8CC'), bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(gridcolor='#233A5E', zerolinecolor='#233A5E', tickfont=dict(color='#A8B8CC')),
    yaxis=dict(gridcolor='#233A5E', zerolinecolor='#233A5E', tickfont=dict(color='#A8B8CC'))
)

# -----------------------------------------------------------------------------
# 3. INTERACTIVE SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/amazon-prime-video.png", width=64)
st.sidebar.title("PrimeView Filters")
st.sidebar.markdown("---")

# Content Type Filter
all_types = sorted(list(df_raw['type'].unique()))
selected_types = st.sidebar.multiselect("Content Type", options=all_types, default=all_types)

# Release Year Filter
min_year = int(df_raw['release_year'].min()) if df_raw['release_year'].notna().any() else 1900
max_year = int(df_raw['release_year'].max()) if df_raw['release_year'].notna().any() else 2026
selected_years = st.sidebar.slider("Release Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# IMDb Rating Filter
min_score = float(df_raw['imdb_score'].min()) if df_raw['imdb_score'].notna().any() else 0.0
max_score = float(df_raw['imdb_score'].max()) if df_raw['imdb_score'].notna().any() else 10.0
selected_score = st.sidebar.slider("IMDb Rating Range", min_value=0.0, max_value=10.0, value=(min_score, max_score), step=0.1)

# Unique Genres Collection
all_genres = sorted(list(set([g for sublist in df_raw['parsed_genres'] for g in sublist])))
selected_genres = st.sidebar.multiselect("Genre", options=all_genres, default=[])

# Unique Countries Collection
all_countries = sorted(list(set([c for sublist in df_raw['parsed_countries'] for c in sublist])))
selected_countries = st.sidebar.multiselect("Production Country", options=all_countries, default=[])

# Age Certification Filter
all_ratings = sorted(list(df_raw['age_certification'].unique()))
selected_ratings = st.sidebar.multiselect("Age Certification", options=all_ratings, default=[])

# -----------------------------------------------------------------------------
# 4. FILTER APPLICATION LOGIC
# -----------------------------------------------------------------------------
df_filtered = df_raw.copy()

if selected_types:
    df_filtered = df_filtered[df_filtered['type'].isin(selected_types)]

df_filtered = df_filtered[
    (df_filtered['release_year'] >= selected_years[0]) & 
    (df_filtered['release_year'] <= selected_years[1])
]

if df_filtered['imdb_score'].notna().any():
    df_filtered = df_filtered[
        (df_filtered['imdb_score'].isna()) | 
        ((df_filtered['imdb_score'] >= selected_score[0]) & (df_filtered['imdb_score'] <= selected_score[1]))
    ]

if selected_genres:
    df_filtered = df_filtered[df_filtered['parsed_genres'].apply(lambda gs: any(g in gs for g in selected_genres))]

if selected_countries:
    df_filtered = df_filtered[df_filtered['parsed_countries'].apply(lambda cs: any(c in cs for c in selected_countries))]

if selected_ratings:
    df_filtered = df_filtered[df_filtered['age_certification'].isin(selected_ratings)]

# -----------------------------------------------------------------------------
# 5. DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">PRIMEVIEW ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Amazon Prime Video — Content Intelligence & Catalogue Insights</div>', unsafe_allow_html=True)
st.markdown(
    "Explore trends, metrics, distributions, and ratings across the Amazon Prime Video streaming library. "
    "Use the interactive sidebar to filter titles dynamically."
)
st.markdown("---")

if df_filtered.empty:
    st.warning("⚠️ No titles match the selected filter criteria. Please broaden your sidebar filters.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. DYNAMIC KPI CARDS
# -----------------------------------------------------------------------------
tot_titles = len(df_filtered)
tot_movies = len(df_filtered[df_filtered['type'] == 'MOVIE'])
tot_shows = len(df_filtered[df_filtered['type'] == 'TV SHOW'])

avg_imdb = df_filtered['imdb_score'].mean()
avg_imdb_str = f"{avg_imdb:.2f}" if pd.notna(avg_imdb) else "N/A"

unique_genres_cnt = len(set([g for sublist in df_filtered['parsed_genres'] for g in sublist]))
unique_countries_cnt = len(set([c for sublist in df_filtered['parsed_countries'] for c in sublist]))

kpi_cols = st.columns(6)

metrics = [
    ("Total Titles", f"{tot_titles:,}"),
    ("Total Movies", f"{tot_movies:,}"),
    ("Total TV Shows", f"{tot_shows:,}"),
    ("Avg IMDb Rating", avg_imdb_str),
    ("Total Genres", f"{unique_genres_cnt}"),
    ("Production Countries", f"{unique_countries_cnt}")
]

for col, (label, val) in zip(kpi_cols, metrics):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Dynamic Automated Insight Box
if tot_titles > 0:
    movie_pct = (tot_movies / tot_titles) * 100
    top_genre_list = [g for sublist in df_filtered['parsed_genres'] for g in sublist]
    top_genre = pd.Series(top_genre_list).mode()[0] if len(top_genre_list) > 0 else "N/A"
    
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>💡 Executive Summary Insight:</strong> The filtered dataset contains <strong>{tot_titles:,} titles</strong>, 
            where Movies represent <strong>{movie_pct:.1f}%</strong> of content. 
            The most prevalent genre in this view is <strong>{top_genre.capitalize()}</strong> with an average IMDb score of <strong>{avg_imdb_str}</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 7. INTERACTIVE VISUALIZATIONS SECTION
# -----------------------------------------------------------------------------
st.markdown("### 📊 Catalogue Intelligence Charts")

row1_col1, row1_col2 = st.columns([1, 2])

# A. Content Distribution (Donut Chart)
with row1_col1:
    st.subheader("Content Distribution")
    type_counts = df_filtered['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    fig_donut = px.pie(
        type_counts, 
        values='Count', 
        names='Type', 
        hole=0.55,
        color='Type',
        color_discrete_map={'MOVIE': '#38BDF8', 'TV SHOW': '#8B7CF6'}
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}: %{value} titles (%{percent})")
    fig_donut.update_layout(**PLOTLY_LAYOUT_DEFAULTS, showlegend=False, height=350)
    st.plotly_chart(fig_donut, use_container_width=True)

# B. Release Year Analysis (Area/Line Chart)
with row1_col2:
    st.subheader("Release Trajectory Over Time")
    year_df = df_filtered.groupby(['release_year', 'type']).size().reset_index(name='count')
    
    fig_line = px.area(
        year_df, 
        x='release_year', 
        y='count', 
        color='type',
        color_discrete_map={'MOVIE': '#38BDF8', 'TV SHOW': '#8B7CF6'},
        labels={'release_year': 'Release Year', 'count': 'Number of Titles', 'type': 'Content Type'}
    )
    fig_line.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

row2_col1, row2_col2 = st.columns(2)

# C. Genre Analysis (Horizontal Bar Chart)
with row2_col1:
    st.subheader("Top 15 Genres")
    genre_series = pd.Series([g.capitalize() for sublist in df_filtered['parsed_genres'] for g in sublist])
    if not genre_series.empty:
        top_genres = genre_series.value_counts().head(15).reset_index()
        top_genres.columns = ['Genre', 'Count']
        
        fig_genre = px.bar(
            top_genres.sort_values(by='Count', ascending=True), 
            x='Count', 
            y='Genre', 
            orientation='h',
            text='Count',
            color_discrete_sequence=['#38BDF8']
        )
        fig_genre.update_traces(textposition='outside', marker_color='#38BDF8')
        fig_genre.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400)
        fig_genre.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_genre, use_container_width=True)
    else:
        st.info("No genre data available for current selection.")

# D. Country Analysis (Horizontal Bar Chart)
with row2_col2:
    st.subheader("Top 15 Production Countries")
    country_series = pd.Series([c for sublist in df_filtered['parsed_countries'] for c in sublist])
    if not country_series.empty:
        top_countries = country_series.value_counts().head(15).reset_index()
        top_countries.columns = ['Country', 'Count']
        
        fig_country = px.bar(
            top_countries.sort_values(by='Count', ascending=True), 
            x='Count', 
            y='Country', 
            orientation='h',
            text='Count',
            color_discrete_sequence=['#8B7CF6']
        )
        fig_country.update_traces(textposition='outside')
        fig_country.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=400)
        fig_country.update_yaxes(autorange="reversed")
    else:
        st.info("No production country data available for current selection.")

st.markdown("---")

row3_col1, row3_col2 = st.columns(2)

# E. IMDb Rating Distribution & Boxplot
with row3_col1:
    st.subheader("IMDb Score Distribution")
    if df_filtered['imdb_score'].notna().any():
        fig_hist = px.histogram(
            df_filtered, 
            x='imdb_score', 
            color='type', 
            barmode='overlay',
            nbins=30,
            color_discrete_map={'MOVIE': '#38BDF8', 'TV SHOW': '#8B7CF6'},
            labels={'imdb_score': 'IMDb Rating'}
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("IMDb rating data missing for current selection.")

with row3_col2:
    st.subheader("IMDb Rating Comparison by Type")
    if df_filtered['imdb_score'].notna().any():
        fig_box = px.box(
            df_filtered, 
            x='type', 
            y='imdb_score', 
            color='type',
            color_discrete_map={'MOVIE': '#38BDF8', 'TV SHOW': '#8B7CF6'},
            points="outliers",
            labels={'imdb_score': 'IMDb Rating', 'type': 'Content Type'}
        )
        fig_box.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=350, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("IMDb rating data missing for current selection.")

# F. Runtime Analysis (Movies Only)
st.markdown("---")
st.subheader("⏱️ Movie Runtime Distribution (Minutes)")
movies_df = df_filtered[(df_filtered['type'] == 'MOVIE') & (df_filtered['runtime'].notna())]

if not movies_df.empty:
    fig_runtime = px.histogram(
        movies_df, 
        x='runtime', 
        nbins=40,
        color_discrete_sequence=['#E5C07B'],
        labels={'runtime': 'Runtime (minutes)'}
    )
    fig_runtime.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=300)
    st.plotly_chart(fig_runtime, use_container_width=True)
else:
    st.info("No movie runtime data available for current selection.")

# -----------------------------------------------------------------------------
# 8. TOP-RATED TITLES LEADERBOARD
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏆 Top-Rated Catalogue Titles")

if 'imdb_votes' in df_filtered.columns and df_filtered['imdb_votes'].notna().any():
    min_vote_threshold = st.slider("Minimum IMDb Vote Count Threshold", min_value=0, max_value=50000, value=1000, step=1000)
    top_rated_df = df_filtered[df_filtered['imdb_votes'] >= min_vote_threshold]
else:
    top_rated_df = df_filtered

if not top_rated_df.empty and 'imdb_score' in top_rated_df.columns:
    display_cols = [c for c in ['title', 'type', 'release_year', 'genres', 'imdb_score', 'imdb_votes', 'age_certification'] if c in top_rated_df.columns]
    top_table = top_rated_df.sort_values(by='imdb_score', ascending=False)[display_cols].head(10)
    
    st.dataframe(
        top_table, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "imdb_score": st.column_config.NumberColumn("IMDb Score", format="%.1f ⭐"),
            "imdb_votes": st.column_config.NumberColumn("Votes", format="%d"),
            "release_year": st.column_config.NumberColumn("Year", format="%d")
        }
    )
else:
    st.info("Unable to calculate top titles with current vote threshold.")

# -----------------------------------------------------------------------------
# 9. SEARCHABLE CATALOGUE & DATA EXPORTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Searchable Catalogue Explorer")

search_term = st.text_input("Search Titles by Name", placeholder="e.g. The Boys, Reacher, Fleabag...")

search_df = df_filtered.copy()
if search_term:
    search_df = search_df[search_df['title'].str.contains(search_term, case=False, na=False)]

st.write(f"Displaying **{len(search_df):,}** matching entries:")

cols_to_show = [c for c in ['title', 'type', 'release_year', 'age_certification', 'runtime', 'imdb_score', 'genres'] if c in search_df.columns]

st.dataframe(
    search_df[cols_to_show], 
    use_container_width=True, 
    hide_index=True
)

# Export Data Feature
csv_data = search_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name="primeview_filtered_catalogue.csv",
    mime="text/csv"
)

# -----------------------------------------------------------------------------
# 10. ABOUT & DATASET QUALITY METRICS
# -----------------------------------------------------------------------------
with st.expander("ℹ️ About PrimeView Analytics & Dataset Quality Summary"):
    st.markdown("""
    **PrimeView Analytics** is an Exploratory Data Analysis (EDA) dashboard engineered to deliver business intelligence metrics 
    and catalogue insights for Amazon Prime Video content.
    """)
    st.markdown("#### Dataset Health Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Dataset Rows", f"{len(df_raw):,}")
    c2.metric("Filtered Dataset Rows", f"{len(df_filtered):,}")
    c3.metric("Missing IMDb Scores", f"{df_raw['imdb_score'].isna().sum():,}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #A8B8CC; font-size: 0.85rem;">'
    'PrimeView Analytics | Built with Python, Streamlit, Pandas & Plotly Graph Objects'
    '</div>', 
    unsafe_allow_html=True
)