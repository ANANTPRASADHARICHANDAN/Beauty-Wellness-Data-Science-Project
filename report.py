import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 1. LOAD & PREP DATA (Unified Pipeline)
print("Loading product_info.csv.zip for final evaluation...")
try:
    df = pd.read_csv('product_info.csv.zip')
except FileNotFoundError:
    print("Error: Upload 'product_info.csv.zip' to Colab.")
    exit()

df = df.dropna(subset=['ingredients', 'price_usd', 'rating']).copy()
df['ingredients'] = df['ingredients'].str.lower()
df['ingredient_count'] = df['ingredients'].apply(lambda x: len(x.split(',')))
df['has_hyaluronic'] = df['ingredients'].apply(lambda x: 1 if 'hyaluronic' in x else 0)
df['has_niacinamide'] = df['ingredients'].apply(lambda x: 1 if 'niacinamide' in x else 0)
df['is_highly_rated'] = df['rating'].apply(lambda x: 1 if x >= 4.5 else 0)

# 2. TRAIN THE FINAL MODEL
X = df[['price_usd', 'ingredient_count', 'has_hyaluronic', 'has_niacinamide']]
y = df['is_highly_rated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 3. VISUAL REPORTING FOR STAKEHOLDERS
print("\n--- Generating Executive Visual Reports ---")

# Visualization 1: Confusion Matrix
# Shows True Positives, False Positives, True Negatives, False Negatives
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Failure', 'Predicted Success'],
            yticklabels=['Actual Failure', 'Actual Success'])
plt.title('Algorithm Performance: Confusion Matrix')
plt.xlabel('Algorithm Prediction')
plt.ylabel('True Market Outcome')
plt.show()

# Visualization 2: Feature Importances
# Shows executives what drives the algorithm's decisions
importances = rf_model.feature_importances_
feature_names = ['Price (USD)', 'Formulation Complexity', 'Hyaluronic Acid', 'Niacinamide']

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('What Drives Product Success? (Feature Importance)')
plt.xlabel('Impact Weight on Customer Satisfaction')
plt.ylabel('Product Attributes')
plt.show()
