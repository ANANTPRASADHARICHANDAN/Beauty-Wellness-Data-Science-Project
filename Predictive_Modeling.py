import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD & PREP DATA
print("Loading product_info.csv.zip...")
try:
    df = pd.read_csv('product_info.csv.zip')
except FileNotFoundError:
    print("Error: Upload 'product_info.csv.zip' to Colab.")
    exit()

df = df.dropna(subset=['ingredients', 'price_usd', 'rating']).copy()
df['ingredients'] = df['ingredients'].str.lower()
df['ingredient_count'] = df['ingredients'].apply(lambda x: len(x.split(',')))
df['has_hyaluronic'] = df['ingredients'].apply(lambda x: 1 if 'hyaluronic' in x else 0)
df['is_highly_rated'] = df['rating'].apply(lambda x: 1 if x >= 4.5 else 0)

# ==========================================
# ALGORITHM 1: RANDOM FOREST CLASSIFIER
# ==========================================
print("\n--- Training Random Forest Classifier ---")
X = df[['price_usd', 'ingredient_count', 'has_hyaluronic']]
y = df['is_highly_rated']

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("\nModel Evaluation (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, zero_division=0))

print("Feature Importances:")
for name, imp in zip(X.columns, rf_model.feature_importances_):
    print(f"- {name}: {imp:.4f}")

# ==========================================
# ALGORITHM 2: CONTENT-BASED FILTERING
# ==========================================
print("\n--- Generating Chemical Cosine Similarity Matrix ---")
# Using a sample of 1000 products to prevent memory crash in standard environments
df_sample = df.head(1000).reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_sample['ingredients'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_chemical_dupes(product_idx, df_reference, sim_matrix):
    sim_scores = list(enumerate(sim_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4] # Get top 3 most similar
    product_indices = [i[0] for i in sim_scores]
    return df_reference['product_name'].iloc[product_indices]

print("\nTop 3 Chemical Dupes for Product 0 (", df_sample['product_name'].iloc[0], "):")
print(get_chemical_dupes(0, df_sample, cosine_sim).to_string(index=False))
