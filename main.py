# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.embedder = None
    
    def fit(self, X, y=None):
        self.embedder = SentenceTransformer(self.model_name)
        return self
    
    def transform(self, X):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)
        
        # Handle different input types
        if hasattr(X, 'values'):  # DataFrame
            texts = X.iloc[:, 0].fillna('').astype(str).tolist()
        elif hasattr(X, 'flatten'):  # numpy array
            texts = [str(text) if text is not None else '' for text in X.flatten()]
        else:  # list or other iterable
            texts = [str(text) if text is not None else '' for text in X]
        
        embeddings = self.embedder.encode(texts)
        return embeddings
    
    def __getstate__(self):
        # Custom pickling - don't save the embedder model, reload it when needed
        state = self.__dict__.copy()
        state['embedder'] = None
        return state
    
    def __setstate__(self, state):
        # Custom unpickling
        self.__dict__.update(state)

# Page Configuration
st.set_page_config(
    page_title="Movies Analysis", 
    page_icon="Assets/movie_icon.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper to load external CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")

# Load CSS
local_css("Assets/style.css")

# --- Load Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Data/Movies.csv")
        # Data cleaning and preprocessing
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date'])
        df['release_year'] = df['release_date'].dt.year
        
        # Handle missing values
        df['vote_average'] = df['vote_average'].fillna(0)
        df['popularity'] = df['popularity'].fillna(0)
        df['vote_count'] = df['vote_count'].fillna(0)
        
        return df
    except FileNotFoundError:
        st.error("Error: 'Movies.csv' not found in the 'Data' directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop()

# --- PAGE CONFIGURATION ---
PAGES = ["Main", "📊 Analysis", "🌍 Languages", "⭐ Highlights", "🤖 Prediction"]

# Get query params (default = Main)
if "page" not in st.query_params:
    st.query_params["page"] = "Main"

current_page = st.query_params["page"]

# --- MAIN PAGE ---
if current_page == "Main":
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            '<div class="animated-title">🎬 Trending Movies Over the Years</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="animated-text">
            Welcome to the Movies Insights Dashboard 🍿
            <br><br>
            Explore how movies have trended across different years, genres, and ratings. 
            Dive into the world of cinema to uncover patterns in popularity, box office hits, 
            and audience preferences.
            <br><br>
            Made By Umesh Pariyar
            <br><br>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Statistics overview
        col_spacer1, col_stat1, col_stat2, col_stat3, col_spacer2 = st.columns([1, 2, 2, 2, 1])
        
        with col_stat1:
            st.metric("Total Movies", f"{len(df):,}")
        with col_stat2:
            st.metric("Languages", len(df['original_language'].unique()))
        with col_stat3:
            st.metric("Year Range", f"{df['release_year'].min()}-{df['release_year'].max()}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Start Analysis", use_container_width=True):
            st.query_params["page"] = "📊 Analysis"
            st.rerun()

# --- DASHBOARD PAGES ---
else:
    # Sidebar navigation
    with st.sidebar:
        st.title("📌 Navigation")
        
        # Page selection
        page = st.selectbox(
            "Select Page:",
            PAGES[1:],
            index=PAGES[1:].index(current_page) if current_page in PAGES[1:] else 0
        )
        
        # Update URL when page changes
        if page != current_page:
            st.query_params["page"] = page
            st.rerun()
        
        st.markdown("---")
        
        if st.button("⬅️ Back to Main", use_container_width=True):
            st.query_params["page"] = "Main"
            st.rerun()

    # --- ANALYSIS PAGE ---
    if page == "📊 Analysis":
        st.title("📊 Movies Analysis")
        
        # Enhanced sidebar filters
        with st.sidebar:
            st.header("🎛️ Filter Options")
            
            # Language filter
            st.subheader("🗣️ Language")
            unique_languages = sorted(df["original_language"].dropna().unique())
            default_index = unique_languages.index("en") if "en" in unique_languages else 0
            selected_language = st.selectbox("Choose a language:", options=unique_languages, index=default_index)
            
            # Vote average filter
            st.subheader("⭐ Rating Range")
            vote_range = st.slider(
                "Average vote range:",
                min_value=float(df["vote_average"].min()),
                max_value=float(df["vote_average"].max()),
                value=(0.0, 10.0),
                step=0.001,
                format="%3f"
            )
            
            # Year range filter
            min_year = int(df['release_year'].min())
            max_year = int(df['release_year'].max())
            st.subheader("📅 Release Year")
            year_range = st.slider(
                "Year range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Adult content filter
            st.subheader("🔞 Content Type")
            adult_cb = st.checkbox("Adult", value=False, key="cb_adult")
            general_cb = st.checkbox("General", value=True, key="cb_general")
            selected_adult = []
            if adult_cb:
                selected_adult.append(True)
            if general_cb:
                selected_adult.append(False)
            if not selected_adult:
                selected_adult = [False]
            
            # Popularity filter
            st.subheader("🔥 Popularity Range")
            pop_range = st.slider(
                "Popularity range:",
                min_value=float(df["popularity"].min()),
                max_value=float(df["popularity"].max()),
                value=(0.0, float(df["popularity"].max())),
                step=0.1
            )

        # Apply filters
        filtered_df = df[
            (df["original_language"] == selected_language) &
            (df["vote_average"] >= vote_range[0]) &
            (df["vote_average"] <= vote_range[1]) &
            (df['release_year'] >= year_range[0]) &
            (df['release_year'] <= year_range[1]) &
            (df['adult'].isin(selected_adult)) &
            (df['popularity'] >= pop_range[0]) &
            (df['popularity'] <= pop_range[1])
        ]
        
        # Display filter results
        st.info(f"Found {len(filtered_df)} movies matching your criteria")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 Timeline", "🔍 Search", "📈 Analytics", "🔥 Trending", "📋 Data"
        ])
        
        with tab1:
            st.subheader("Movies Released Over Time")
            if not filtered_df.empty:
                movie_count = filtered_df.groupby('release_year').size().reset_index(name='count')
                
                # Create interactive plotly chart
                fig = px.line(movie_count, x='release_year', y='count', 
                            title=f"Movies per Year ({selected_language})")
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Year",
                    yaxis_title="Number of Movies"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for selected filters")
        
        with tab2:
            st.subheader("🔍 Movie Search")
            search_title = st.text_input("Search by movie title:")
            
            if search_title and not filtered_df.empty:
                results = filtered_df[
                    filtered_df['title'].str.contains(search_title, case=False, na=False)
                ]
                
                if not results.empty:
                    st.dataframe(
                        results[['title', 'release_year', 'vote_average', 'popularity']],
                        use_container_width=True
                    )
                else:
                    st.warning("No movies found matching your search")
            elif search_title:
                st.warning("No data available for selected filters")
        
        with tab3:
            st.subheader("📈 Vote Count vs Popularity")
            if not filtered_df.empty and len(filtered_df) > 1:
                fig = px.scatter(
                    filtered_df, 
                    x='vote_count', 
                    y='popularity',
                    color='adult',
                    hover_data=['title', 'release_year'],
                    title="Popularity vs Vote Count",
                    color_discrete_map={True: '#ff6b6b', False: '#4ecdc4'}
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for scatter plot")
        
        with tab4:
            st.subheader("🔥 Most Popular Movies")
            if not filtered_df.empty:
                top_popular = filtered_df.nlargest(10, 'popularity')
                
                fig = px.bar(
                    top_popular, 
                    x='popularity', 
                    y='title',
                    orientation='h',
                    title="Top 10 Most Popular Movies"
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available")
        
        with tab5:
            st.subheader("📋 Filtered Dataset")
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name=f"filtered_movies.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data matches current filters")

    # --- LANGUAGES PAGE ---
    elif page == "🌍 Languages":
        st.title("🌍 Language Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Popularity", "📈 Trends", "⭐ Ratings"])
        
        with tab1:
            st.subheader("Average Popularity by Language")
            lang_popularity = df.groupby('original_language')['popularity'].mean().sort_values(ascending=False).head(15)
            
            fig = px.bar(
                x=lang_popularity.values,
                y=lang_popularity.index,
                orientation='h',
                title="Top 15 Languages by Average Popularity"
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Popularity Trends Over Time")
            # Get top 5 languages by movie count
            top_languages = df['original_language'].value_counts().head(5).index.tolist()
            lang_trend = df[df['original_language'].isin(top_languages)]
            
            lang_trend_agg = lang_trend.groupby(['release_year', 'original_language'])['popularity'].mean().reset_index()
            
            fig = px.line(
                lang_trend_agg,
                x='release_year',
                y='popularity',
                color='original_language',
                title="Popularity Trends for Top 5 Languages"
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Highest Rated Languages")
            lang_ratings = df.groupby('original_language').agg({
                'vote_average': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'movie_count'})
            
            # Filter languages with at least 10 movies
            lang_ratings = lang_ratings[lang_ratings['movie_count'] >= 10]
            lang_ratings = lang_ratings.sort_values('vote_average', ascending=False).head(15)
            
            fig = px.bar(
                x=lang_ratings['vote_average'],
                y=lang_ratings.index,
                orientation='h',
                title="Top Rated Languages (≥10 movies)"
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)

    # --- HIGHLIGHTS PAGE ---
    elif page == "⭐ Highlights":
        st.title("⭐ Movie Highlights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Rated Movies")
            # Filter movies with minimum votes for credibility
            top_rated = df[df['vote_count'] >= 100].nlargest(10, 'vote_average')
            
            for idx, movie in top_rated.iterrows():
                st.metric(
                    movie['title'][:30] + "..." if len(movie['title']) > 30 else movie['title'],
                    f"⭐ {movie['vote_average']}/10",
                    f"📅 {int(movie['release_year'])}"
                )
        
        with col2:
            st.subheader("🔥 Most Popular Movies")
            most_popular = df.nlargest(10, 'popularity')
            
            for idx, movie in most_popular.iterrows():
                st.metric(
                    movie['title'][:30] + "..." if len(movie['title']) > 30 else movie['title'],
                    f"🔥 {movie['popularity']:.1f}",
                    f"⭐ {movie['vote_average']}/10"
                )
        
        # Additional insights
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            best_year = df.groupby('release_year')['vote_average'].mean().idxmax()
            st.metric("Best Year (Avg Rating)", int(best_year), f"{df.groupby('release_year')['vote_average'].mean().max():.2f}")
        
        with col4:
            most_productive_year = df['release_year'].mode()[0]
            movie_count = len(df[df['release_year'] == most_productive_year])
            st.metric("Most Productive Year", int(most_productive_year), f"{movie_count} movies")
        
        with col5:
            top_language = df['original_language'].mode()[0]
            lang_count = len(df[df['original_language'] == top_language])
            st.metric("Top Language", top_language.upper(), f"{lang_count} movies")
        
        with col6:
            adult_percentage = (df['adult'].sum() / len(df)) * 100
            st.metric("Adult Content %", f"{adult_percentage:.1f}%")

    # --- PREDICTION PAGE ---
    elif page == "🤖 Prediction":
        st.title("🤖 Movie Rating Prediction")

        @st.cache_resource
        def load_model():
            try:
                return joblib.load("notebooks/vote_average_predictor.pkl")
            except FileNotFoundError:
                st.error("Model file 'vote_average_predictor.pkl' not found. Please retrain the model.")
                return None
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None

        model = load_model()

        if model is not None:
            st.info("This model is used to predict movie's vote average based on title, overview, language, content type, and popularity.")

            with st.form("prediction_form"):
                st.subheader("Movie Details")
                
                # Use The Shawshank Redemption as default (from your actual dataset)
                title = st.text_input("Movie Title", value="The Shawshank Redemption")
                
                # Get the actual overview from your dataset
                shawshank_overview = "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an old con named Red -- for his integrity and unquenchable sense of hope."
                
                overview = st.text_area("Movie Overview/Plot", 
                                    value=shawshank_overview, 
                                    height=150)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    unique_languages = sorted(df["original_language"].dropna().unique())
                    original_language = st.selectbox("Original Language", 
                                                options=unique_languages, 
                                                index=unique_languages.index("en") if "en" in unique_languages else 0)
                    
                with col2:
                    adult = st.selectbox("Content Rating", 
                                    options=[False, True], 
                                    format_func=lambda x: "General Audience" if not x else "Adult Content")
                
                # Use actual popularity from your dataset
                popularity = st.slider("Expected Popularity Score", 
                                    min_value=float(df["popularity"].min()),
                                    max_value=float(df["popularity"].max()),
                                    value=35.7248,  # Actual Shawshank popularity from your dataset
                                    step=0.0001,
                                    format="%4f"
                                    )

                submitted = st.form_submit_button("Predict Rating")

            if submitted:
                if not title.strip() or not overview.strip():
                    st.error("Please enter both movie title and overview")
                else:
                    input_data = pd.DataFrame([{
                        "title": title.strip(),
                        "overview": overview.strip(),
                        "original_language": original_language,
                        "adult": adult,
                        "popularity": popularity
                    }])

                    try:
                        with st.spinner("Generating NLP embeddings and predicting..."):
                            prediction = model.predict(input_data)[0]
                        
                        # Display prediction with actual comparison
                        st.metric("Predicted Rating", f"{prediction:.2f}/10")
                        
                        # Add interpretation
                        if prediction >= 8.0:
                            st.balloons()
                            st.success("🏆 Excellent! This movie is predicted to be highly rated!")
                        elif prediction >= 7.0:
                            st.info("👍 Good! This movie should receive positive reviews.")
                        elif prediction >= 6.0:
                            st.info("👌 Decent rating expected.")
                        elif prediction >= 5.0:
                            st.warning("😐 Average rating expected.")
                        else:
                            st.warning("👎 Below average rating predicted.")
                            
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
            
            # Example movies section from your actual dataset
            st.markdown("---")
            st.subheader("Example Movies from Dataset")
            
            examples = [
                {
                    "title": "The Godfather",
                    "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers, launching a campaign of bloody revenge.",
                    "rating": "8.687", 
                    "popularity": "31.7735"
                },
                {
                    "title": "Lilo & Stitch",
                    "overview": "The wildly funny and touching story of a lonely Hawaiian girl and the fugitive alien who helps to mend her broken family.",
                    "rating": "7.063",
                    "popularity": "508.3123"
                },
                {
                    "title": "Warfare",
                    "overview": "A platoon of Navy SEALs embarks on a dangerous mission in Ramadi, Iraq, with the chaos and brotherhood of war retold through their memories of the event.",
                    "rating": "7.226",
                    "popularity": "262.0183"
                }
            ]
            
            for i, example in enumerate(examples):
                with st.expander(f"Example {i+1}: {example['title']} (Rating: {example['rating']})"):
                    st.write(f"**Overview:** {example['overview']}")
                    st.write(f"**Actual Rating:** {example['rating']}/10")
                    st.write(f"**Popularity:** {example['popularity']}")
                    
        else:
            st.error("Model could not be loaded.")