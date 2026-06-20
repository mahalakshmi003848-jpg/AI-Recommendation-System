import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("courses.csv")

# User interests
user_input = input("Enter your interests (comma separated): ")

# Combine user profile with course skills
documents = [user_input] + data["Skills"].tolist()

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
).flatten()

# Add scores
data["Similarity"] = similarity_scores

# Top recommendations
recommendations = data.sort_values(
    by="Similarity",
    ascending=False
)

print("\nTop Recommendations:\n")
for _, row in recommendations.head(3).iterrows():
    print(
        f"{row['Course']} "
        f"(Match Score: {row['Similarity']:.2f})"
    )