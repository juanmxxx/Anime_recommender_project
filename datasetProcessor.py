import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('dataset/small2.csv')



# Ensure 'Genres' and 'Synopsis' columns have no NaN values
df['Genres'] = df['Genres'].fillna('')
df['Synopsis'] = df['Synopsis'].fillna('')

# Combine 'Genres' and 'Synopsis' into a single text column
df['combined_features'] = df['Genres'] + " " + df['Synopsis']

# Wanna have into account the Score and Rank
# Normalize the Score and Rank columns
df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce').fillna(0)

# Normalize the Score and Rank columns to a range of 0 to 1
df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min())
df['Rank'] = (df['Rank'] - df['Rank'].min()) / (df['Rank'].max() - df['Rank'].min())
# Combine 'Genres', 'Synopsis', 'Score', and 'Rank' into a single text column
df['combined_features'] = df['combined_features'] + " " + df['Score'].astype(str) + " " + df['Rank'].astype(str)

# Drop the original 'Genres' and 'Synopsis' columns
df = df.drop(columns=['Genres', 'Synopsis'])
# Drop the 'anime_id' column
df = df.drop(columns=['anime_id'])

# Now vectorize the combined features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Function to recommend animes based on keywords
def recommend_anime(keywords, top_n=10):
    # Vectorize the input keywords
    keywords_vector = tfidf_vectorizer.transform([keywords])

    # Compute cosine similarity between the input and all animes
    similarity_scores = cosine_similarity(keywords_vector, tfidf_matrix)

    # Add similarity scores to the DataFrame
    df['similarity'] = similarity_scores[0]

    # Sort by similarity, Score, and Rank
    recommendations = df.sort_values(by=['similarity', 'Score', 'Rank'], ascending=[False, False, True])

    # Process the Score and Rank columns for make legible to the user
    recommendations['Score'] = recommendations['Score'].apply(lambda x: round(x * 10, 1))  # Scale back to original score

    # Return the rank to the original rank
    recommendations['Rank'] = recommendations['Rank'].apply(lambda x: round(x * 100))  # Scale back to original rank

    # Return the top N animes
    return recommendations[['Name', 'Score', 'Rank', 'similarity']].head(top_n)

# Example usage
keywords = "romance comedy love story"
recommended_animes = recommend_anime(keywords, top_n=5)
print(recommended_animes)



#
#
#
#
#
# # Vectorize the combined features using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
#
# # Function to recommend animes based on keywords
# def recommend_anime(keywords, top_n=5):
#     # Vectorize the input keywords
#     keywords_vector = tfidf_vectorizer.transform([keywords])
#
#     # Compute cosine similarity between the input and all animes
#     similarity_scores = cosine_similarity(keywords_vector, tfidf_matrix)
#
#     # Get the indices of the top N most similar animes
#     top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
#
#     # Return the top N animes
#     return df.iloc[top_indices][['Name', 'Genres', 'Synopsis', 'Score']]
#
# # Example usage
# keywords = "superhero action adventure"
# recommended_animes = recommend_anime(keywords, top_n=5)
# print(recommended_animes)
#
#
#
