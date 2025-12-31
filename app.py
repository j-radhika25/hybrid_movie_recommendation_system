import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MOVIE RECOMMENDER",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
/* Center the title */
h1 {
    text-align: center;
}
/* Style the primary button */
.stButton > button {
    color: white;
    background-color: #0068C9; /* A nice shade of blue */
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
}
.stButton > button:hover {
    background-color: #0055A4; /* Darker blue on hover */
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_similarity():
    """Loads dataset, cleans it, and computes similarity matrix."""
    try:
        df = pd.read_csv('imdb_top_1000.csv')
        
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
        df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)
        df['Gross'] = df['Gross'].str.replace(',', '', regex=False).astype(float)
        df['Released_Year'].fillna(df['Released_Year'].median(), inplace=True)
        df['Runtime'].fillna(df['Runtime'].median(), inplace=True)
        df['Gross'].fillna(df['Gross'].median(), inplace=True)
        df['Meta_score'].fillna(df['Meta_score'].median(), inplace=True)
        df['Certificate'].fillna(df['Certificate'].mode()[0], inplace=True)
        df['Primary_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])
        df['soup'] = df['Primary_Genre'] + ' ' + df['Director'] + ' ' + df['Star1']
        df['soup'] = df['soup'].fillna('')
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        return df, cosine_sim
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except FileNotFoundError:
        return None

df, cosine_sim = load_data_and_similarity()
model = load_model()

if df is not None:
    indices = pd.Series(df.index, index=df['Series_Title']).drop_duplicates()
    features = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross', 'Primary_Genre']

def get_recommendations(title, model, cosine_sim_matrix):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        similar_movies = df.iloc[movie_indices]
        recommendation_features = similar_movies[features]
        quality_probs = model.predict_proba(recommendation_features)[:, 1]
        final_recs = similar_movies.copy()
        final_recs['predicted_quality'] = quality_probs
        final_recs = final_recs.sort_values(by=['predicted_quality'], ascending=False)
        return final_recs
    except (KeyError, IndexError):
        return pd.DataFrame()

st.markdown("<h1>🎬 MOVIE RECOMMENDER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover your next favorite movie! Select a title, and our AI will recommend similar movies, filtered for the highest quality.</p>", unsafe_allow_html=True)
st.markdown("---")

if df is not None and model is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        movie_list = df['Series_Title'].sort_values().unique()
        selected_movie = st.selectbox(
            "Choose a movie you like",
            movie_list
        )
        
        b_col1, b_col2, b_col3 = st.columns([1, 2.3, 1])
        with b_col2:
            get_recs_button = st.button('Get Recommendations', type="primary", use_container_width=True)

    if get_recs_button:
        with st.spinner(f'Analyzing recommendations for "{selected_movie}"...'):
            recommendations = get_recommendations(selected_movie, model, cosine_sim)

            if not recommendations.empty:
                st.markdown("---")
                st.subheader(f"Top 5 Recommendations based on '{selected_movie}':")

                for i, (index, row) in enumerate(recommendations.head(5).iterrows()):
                    st.markdown(f"### **{i+1}. {row['Series_Title']}** ({int(row['Released_Year'])})")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**IMDb Rating:** ⭐ {row['IMDB_Rating']}/10")
                        st.markdown(f"**Predicted Quality Score:** {row['predicted_quality']:.2f}")
                        st.markdown(f"**Runtime:** {row['Runtime']} min")
                    with c2:
                        st.markdown(f"**Genre:** {row['Genre']}")
                        st.markdown(f"**Certificate:** {row['Certificate']}")
                        st.markdown(f"**Director:** {row['Director']}")

                    st.markdown(f"**Stars:** {row['Star1']}, {row['Star2']}, {row['Star3']}")
                    st.markdown(f"**Overview:** {row['Overview']}")
                    st.markdown("---")
            else:
                st.error("Sorry, could not generate recommendations for this movie.")
else:
    st.error("Some critical files are missing!")