#Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movies Analysis", layout="wide")

#--- SESSION STATE FOR NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "home"  # default page

#--- HOME PAGE ---
if st.session_state.page == "home":
    st.title("💼 Data Science Jobs Salary Analysis")
    st.subheader("Welcome to the Salary Insights Dashboard 📊")
    st.markdown(
        """
        Discover trends, analyze patterns, and explore salaries across different roles, 
        experience levels, and company sizes in the data science industry.

        Made by Chirag Sharma

        🚀 Click below to start your journey!
        """
    )

    if st.button("👉 Start Analysis"):
        st.session_state.page = "analysis"
        st.rerun()

# ---Analysis Page---
elif st.session_state.page == "analysis":
    # Load Dataset
    try:
        df = pd.read_csv("Data/Movies.csv")
    except FileNotFoundError:
        st.error("Error: 'Movies.csv' not found in the 'data' directory.")
        st.stop()

    # st.markdown("## Sample Data")
    # st.write(df.head())

    #st.write("Python")
    #st.write(["Python", "JS", "Java"])
    #st.divider()
    #st.dataframe(df.head())

    #st.markdown("##### Machine Learning Engineer")

    # def filter_original_language(df, column_value:str):
    #     custom_filter = df["original_language"] == column_value
    #     filtered_df = df[custom_filter]
    #     return filtered_df

    # #selected_column_value = "en"
    # selected_column_value = st.text_input("Original Language","en")
    # if selected_column_value:
    #     filtered_df = filter_original_language(df, selected_column_value)
    #     st.dataframe(filtered_df)


    ## Sidebar
    with st.sidebar:
        st.header("Filter Options")
        st.title("Language Option")
        unique_languages = df["original_language"].dropna().unique()
        selected_language = st.selectbox(
            "Choose a language:",
            options=unique_languages,
            index=0,
            placeholder="Select Language",
        )

        # Slider
        st.title("Average Vote Filter")
        st.subheader("Slider Selector:")
        vote_range = st.slider("Select average vote range:", min_value = df["vote_average"].min(), max_value = df["vote_average"].max(), value=(0.000, 10.000),
        step=0.001, format="%.3f")
        st.write("Selected range:", vote_range)

        #Slider 2
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        df = df.dropna(subset=['release_date'])

        df['release_year'] = df['release_date'].dt.year

        min_year = int(df['release_year'].min())
        max_year = int(df['release_year'].max())

        st.sidebar.header("Adult Genre and Year Filter")

        year_range = st.sidebar.slider(
            "Select Release Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        selected_adult = st.sidebar.multiselect(
            "Include Adult Content?",
            options=[True, False],
            default=[False]
        )

        
    st.markdown("## 🎬Main Filtered Values!!!")
    if "vote_average" and "original_language" and "release_year" in df.columns:
        filtered_df = df[
            (df["original_language"] == selected_language) &    
            (df["vote_average"] >= vote_range[0]) &
            (df["vote_average"] <= vote_range[1]) &
            (df['release_year'] >= year_range[0]) &
            (df['release_year'] <= year_range[1]) &
            (df['adult'].isin(selected_adult))
    ]
        st.dataframe(filtered_df)

    else:
        st.error("Column 'vote_average' or 'original_language' or 'release_year' not found in the dataset.")

    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.divider()

    st.markdown("## ANALYSING TABS FOR FILTERED VALUES")

    ## Tabs

    tab1, tab2, tab3, tab4= st.tabs(["Movies Per Year", "Movie Information", "Popularity vs Vote Count", "Trending Movies"])
    with tab1:
        st.subheader("🎬 Number of Movies Released per Year")
        movie_count_by_year = filtered_df.groupby('release_year')['id'].count().reset_index()
        movie_count_by_year.columns = ['release_year', 'movie_count']
        st.bar_chart(movie_count_by_year.set_index('release_year'))

    with tab2:
        st.subheader("🎬 Search the name of Movie for Information:")
        search_title = st.text_input("🔍 Search Movie Title:")
        if search_title:
            search_results = filtered_df[filtered_df['title'].str.contains(search_title, case=False, na=False)]
            st.write(f"Search Results for: {search_title}")
            st.dataframe(search_results[['id','title', 'release_year', 'vote_average', 'popularity', 'release_date']])

    with tab3:
        st.subheader("📈 Popularity vs. Vote Count")
        fig, ax = plt.subplots()
        ax.scatter(filtered_df['vote_count'], filtered_df['popularity'], alpha=0.5, c=filtered_df['adult'].map({True: 'red', False: 'blue'}))
        ax.set_xlabel("Vote Count")
        ax.set_ylabel("Popularity")
        ax.set_title("Popularity vs. Vote Count")
        ax.grid(True)
        st.pyplot(fig)

    with tab4:
        st.subheader("🔥Top 10 Trending Movies (by Popularity)")
        top_popular = filtered_df.sort_values("popularity", ascending=False).head(10)
        st.dataframe(top_popular[['title', 'release_year', 'popularity', 'vote_average', 'vote_count']])

    st.divider()

    st.markdown("## ANALYSING TABS FOR ALL DATASET")

    ## Tabs

    tab1, tab2, tab3, tab4= st.tabs(["Movies Per Year", "Movie Information", "Popularity vs Vote Count", "Trending Movies"])
    with tab1:
        st.subheader("🎬 Number of Movies Released per Year")
        movie_count_by_year1 = df.groupby('release_year')['id'].count().reset_index()
        movie_count_by_year1.columns = ['release_year', 'movie_count']
        st.bar_chart(movie_count_by_year1.set_index('release_year'))

    with tab2:
        st.subheader("🎬 Search the name of Movie for Information:")
        search_title1 = st.text_input("🔍 Search for Movie Title:")
        if search_title1:
            search_results1 = df[df['title'].str.contains(search_title1, case=False, na=False)]
            st.write(f"Search Results for: {search_title1}")
            st.dataframe(search_results1[['id','title', 'release_year', 'vote_average', 'popularity', 'release_date']])

    with tab3:
        st.subheader("📈 Popularity vs. Vote Count")
        fig, ax1 = plt.subplots()
        ax1.scatter(df['vote_count'], df['popularity'], alpha=0.5, c=df['adult'].map({True: 'red', False: 'blue'}))
        ax1.set_xlabel("Vote Count")
        ax1.set_ylabel("Popularity")
        ax1.set_title("Popularity vs. Vote Count")
        ax1.grid(True)
        st.pyplot(fig)

    with tab4:
        st.subheader("🔥Top 10 Trending Movies (by Popularity)")
        top_popular1 = df.sort_values("popularity", ascending=False).head(10)
        st.dataframe(top_popular1[['title', 'release_year', 'popularity', 'vote_average', 'vote_count']])


    st.divider()
    st.markdown("## ANALYZING LANGUAGES")


    tab1, tab2, tab3= st.tabs(["Highest Popularity", "Highest Popularity Over The Years", "Highest Average Ratings"])

    with tab1:
        st.subheader("🌍 Languages with Highest Popularity")
        lang_counts = df.groupby('original_language')['popularity'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(lang_counts)

    with tab2:
        st.subheader("🌍 Languages with Highest Popularity over the years")
        lang_trend = df.groupby(['release_year', 'original_language'])['popularity'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=lang_trend, x='release_year', y='popularity', hue='original_language')
        plt.title("Average Popularity by Language Over Time")
        st.pyplot(plt)

    with tab3:
        top_lang_rating = df.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).head(10)
        st.subheader("🌍 Languages with Highest Average Ratings")
        st.bar_chart(top_lang_rating)

    st.divider()

    st.markdown("## VIEWING COLUMNS")
    ## COLUMNS

    col1, col2 = st.columns([3 , 3])
    with col1:
        st.markdown("### 🔥Top 5 Most Rated Movies")
        top_movies = df.groupby("title", as_index=False)["vote_average"].mean()
        top_movies = top_movies.sort_values(by="vote_average", ascending=False).head(5)
        st.dataframe(top_movies)

    with col2:
        st.markdown("### 🔥Top 5 Most Rated Movies")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(top_movies["title"],top_movies["vote_average"])
        ax.set_xlabel("Title")
        ax.set_ylabel("Average Vote")
        st.pyplot(fig)




