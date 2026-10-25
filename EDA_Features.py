import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load the dataset natively from the ZIP file
print("Loading product_info.csv.zip...")
try:
    df = pd.read_csv('product_info.csv.zip')
    print(f"Dataset loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
except FileNotFoundError:
    print("Error: Upload 'product_info.csv.zip' to your Colab session first.")
    exit()

# 2. FEATURE ENGINEERING
print("\n--- Engineering New Features ---")
if 'ingredients' in df.columns:
    df['ingredients'] = df['ingredients'].fillna('').str.lower()
    
    # Create interaction and count features for the model
    df['ingredient_count'] = df['ingredients'].apply(lambda x: len(x.split(',')) if x else 0)
    df['has_hyaluronic_acid'] = df['ingredients'].apply(lambda x: 1 if 'hyaluronic' in x else 0)
    
if 'rating' in df.columns:
    # Binary target transformation for Week 5 classification
    df['is_highly_rated'] = df['rating'].apply(lambda x: 1 if pd.notnull(x) and x >= 4.5 else 0)

print("Engineered successfully: 'ingredient_count', 'has_hyaluronic_acid', 'is_highly_rated'")

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- Running Exploratory Data Analysis ---")
if 'price_usd' in df.columns:
    # Visualization 1: Distribution of prices
    plt.figure(figsize=(8, 5))
    sns.histplot(df['price_usd'].dropna(), bins=50, kde=True, color='teal')
    plt.title('Distribution of Product Prices (USD)')
    plt.xlabel('Price ($)')
    plt.xlim(0, 300) # Capping extreme outliers for a readable chart
    plt.show()

    # Visualization 2: Price variance by active ingredient
    if 'has_hyaluronic_acid' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='has_hyaluronic_acid', y='price_usd', data=df, palette='Set2')
        plt.title('Product Price vs. Hyaluronic Acid Presence')
        plt.xlabel('Contains Hyaluronic Acid (0 = No, 1 = Yes)')
        plt.ylabel('Price (USD)')
        plt.ylim(0, 150)
        plt.show()

# 4. STATISTICAL HYPOTHESIS TESTING
if 'price_usd' in df.columns and 'has_hyaluronic_acid' in df.columns:
    print("\n--- Statistical Hypothesis Testing ---")
    group_yes = df[df['has_hyaluronic_acid'] == 1]['price_usd'].dropna()
    group_no = df[df['has_hyaluronic_acid'] == 0]['price_usd'].dropna()
    
    # Independent T-Test to prove our Week 1 hypotheses mathematically
    t_stat, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
    print(f"T-Statistic: {t_stat:.4f} | P-Value: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Business Insight: There is a statistically significant price premium for Hyaluronic Acid.")
    else:
        print("Business Insight: No significant price difference found for this ingredient.")
