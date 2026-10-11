import pandas as pd
import numpy as np
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler

# Configure Reproducible Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SkincarePreprocessingPipeline:
    """
    End-to-End Data Cleaning & Preprocessing Pipeline for Beauty E-Commerce Data.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'} # Keep negations for sentiment!
        self.lemmatizer = WordNetLemmatizer()
        self.scaler = MinMaxScaler()
        
        # Chemical Synonym Dictionary
        self.chemical_synonyms = {
            r'\b(l-ascorbic acid|ascorbic acid|ascorbyl glucoside)\b': 'vitamin c',
            r'\b(retinyl palmitate|retinaldehyde|adapalene)\b': 'retinol',
            r'\b(nicotinamide|vitamin b3)\b': 'niacinamide',
            r'\b(sodium hyaluronate|hydrolyzed hyaluronic acid)\b': 'hyaluronic acid',
            r'\b(parfum|linalool|limonene|citronellol)\b': 'fragrance'
        }

    def handle_missing_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Initial dataset shape: {df.shape}")
        
        # 1. Drop duplicate reviews from the same user on the same product
        df = df.drop_duplicates(subset=['author_id', 'product_id'], keep='first').copy()
        
        # 2. Strict Deletion for critical unstructured columns
        df = df.dropna(subset=['review_text', 'ingredients', 'star_rating']).copy()
        
        # 3. Categorical Imputation for Skin Type
        df['skin_type'] = df['skin_type'].fillna('Unspecified').str.title()
        
        # 4. Median Imputation for Price grouped by Category
        if 'price_usd' in df.columns and 'category' in df.columns:
            df['price_usd'] = df.groupby('category')['price_usd'].transform(
                lambda x: x.fillna(x.median())
            )
        logging.info(f"Shape after missing/duplicate handling: {df.shape}")
        return df

    def clean_review_text(self, text: str) -> str:
        # Lowercase and strip HTML tags
        text = str(text).lower()
        text = re.sub(r'<.*?>', ' ', text)
        # Keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize, remove stopwords (keeping 'not'/'no'), and Lemmatize
        words = text.split()
        cleaned = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return ' '.join(cleaned)

    def standardize_ingredients(self, ing_string: str) -> str:
        ing_text = str(ing_string).lower()
        # Replace chemical synonyms with canonical names
        for pattern, canonical in self.chemical_synonyms.items():
            ing_text = re.sub(pattern, canonical, ing_text)
        return ing_text

    def handle_outliers_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Filter out low-signal reviews (< 3 words after cleaning)
        df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= 3].copy()
        
        # 2. Log-transform right-skewed price data & MinMax Scale [0, 1]
        df['price_log'] = np.log1p(df['price_usd'])
        df[['price_normalized', 'rating_normalized']] = self.scaler.fit_transform(
            df[['price_log', 'star_rating']]
        )
        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Skincare Preprocessing Pipeline...")
        df = self.handle_missing_and_duplicates(df)
        
        logging.info("Executing NLP Text Cleaning & Lemmatization...")
        df['cleaned_review'] = df['review_text'].apply(self.clean_review_text)
        
        logging.info("Standardizing Chemical Ingredient Nomenclature...")
        df['standardized_ingredients'] = df['ingredients'].apply(self.standardize_ingredients)
        
        # Engineer binary flags for top dermatological actives
        for active in ['retinol', 'vitamin c', 'niacinamide', 'hyaluronic acid', 'salicylic acid', 'fragrance']:
            col_name = f"has_{active.replace(' ', '_')}"
            df[col_name] = df['standardized_ingredients'].apply(lambda x: 1 if active in x else 0)
            
        logging.info("Applying Outlier Filtering and Min-Max Normalization...")
        df = self.handle_outliers_and_scale(df)
        
        # One-Hot Encode Skin Type
        df = pd.get_dummies(df, columns=['skin_type'], prefix='skin', dtype=int)
        
        logging.info(f"Pipeline Complete! Final dataset shape: {df.shape}")
        return df

# --- SIMULATION TEST WITH SAMPLE RAW MESSY DATA ---
if __name__ == "__main__":
    raw_sample_data = {
        'product_id': ['P01', 'P01', 'P02', 'P03', 'P04'],
        'author_id': ['U1', 'U1', 'U2', 'U3', 'U4'], # U1 is a duplicate!
        'category': ['Serum', 'Serum', 'Moisturizer', 'Cleanser', 'Serum'],
        'price_usd': [45.0, 45.0, 320.0, np.nan, 18.0], # 320 is an outlier, P03 is missing price!
        'skin_type': ['Oily', 'Oily', np.nan, 'Dry', 'Sensitive'], # U2 is missing skin_type!
        'star_rating': [5, 5, 1, 4, 2],
        'review_text': [
            "<b>OMG!</b> This serum hydrated my skin so well!! Best ever.",
            "<b>OMG!</b> This serum hydrated my skin so well!! Best ever.",
            "Did not work! Caused massive breakouts and burning.",
            "Good.", # Outlier: Too short (< 3 words) -> Should be filtered!
            "Smells nice, but made my face super red and itchy."
        ],
        'ingredients': [
            "Aqua, L-Ascorbic Acid, Sodium Hyaluronate, Glycerin",
            "Aqua, L-Ascorbic Acid, Sodium Hyaluronate, Glycerin",
            "Water, Retinyl Palmitate, Parfum, Linalool",
            "Water, Ceramide NP, Glycerin",
            "Aqua, Nicotinamide, Salicylic Acid, Limonene"
        ]
    }
    
    raw_df = pd.DataFrame(raw_sample_data)
    pipeline = SkincarePreprocessingPipeline()
    cleaned_df = pipeline.run_pipeline(raw_df)
    print("\n--- CLEANED FEATURE MATRIX PREVIEW ---")
    print(cleaned_df[['product_id', 'cleaned_review', 'has_vitamin_c', 'has_fragrance', 'price_normalized']])
