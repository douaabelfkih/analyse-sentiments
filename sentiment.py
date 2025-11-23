import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour éviter les blocages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Télécharger les ressources NLTK nécessaires
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# ==================== 1. CHARGEMENT DES DONNÉES ====================
def load_data(filepath):
    """Charge le dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

# ==================== 2. EXPLORATION DES DONNÉES ====================
def explore_data(df):
    """Affiche des statistiques descriptives"""
    print("\n=== APERÇU DES DONNÉES ===")
    print(df.head())
    
    print("\n=== INFORMATIONS SUR LES COLONNES ===")
    print(df.info())
    
    print("\n=== VALEURS MANQUANTES ===")
    print(df.isnull().sum())
    
    print("\n=== DISTRIBUTION DES SCORES ===")
    print(df['Score'].value_counts().sort_index())
    
    # Visualisation
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    df['Score'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution des Scores')
    plt.xlabel('Score')
    plt.ylabel('Nombre d\'avis')
    
    plt.subplot(1, 2, 2)
    df['Text'].str.len().hist(bins=50, color='coral')
    plt.title('Distribution de la longueur des avis')
    plt.xlabel('Longueur du texte')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig('exploration_data.png', dpi=100, bbox_inches='tight')
    print("📊 Graphique sauvegardé : exploration_data.png")
    plt.close()

# ==================== 3. PREPROCESSING ====================
def preprocess_text(text):
    """Nettoie et prétraite le texte"""
    if pd.isna(text):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression des mentions et hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Suppression des caractères spéciaux et chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_sentiment_label(score, method='binary'):
    """
    Crée les labels de sentiment
    method: 'binary' (positif/négatif), 'ternary' (positif/neutre/négatif), ou 'multiclass' (5 classes)
    """
    if method == 'binary':
        return 1 if score > 3 else 0  # 1=Positif, 0=Négatif
    elif method == 'ternary':
        if score <= 2:
            return 0  # Négatif
        elif score == 3:
            return 1  # Neutre
        else:
            return 2  # Positif
    else:  # multiclass
        return score - 1  # 0-4 pour classification multiclasse

def prepare_data(df, method='binary', sample_size=None):
    """Prépare les données pour le modèle"""
    # Copie pour éviter les modifications du dataframe original
    df = df.copy()
    
    # Échantillonnage si nécessaire (pour tests rapides)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Suppression des valeurs manquantes
    df = df.dropna(subset=['Text', 'Score'])
    
    # Combinaison du Summary et Text pour plus d'information
    df['combined_text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
    
    # Preprocessing du texte
    print("Preprocessing du texte en cours...")
    df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
    
    # Création des labels
    df['sentiment'] = df['Score'].apply(lambda x: create_sentiment_label(x, method))
    
    # Features additionnelles
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
    
    print(f"Données préparées : {len(df)} avis")
    print(f"Distribution des sentiments : \n{df['sentiment'].value_counts()}")
    
    return df

# ==================== 4. CRÉATION DES FEATURES ====================
def create_features(X_train, X_test, max_features=5000):
    """Crée les features TF-IDF"""
    print(f"Création des features TF-IDF (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrammes et bigrammes
        min_df=5,  # Ignore les mots apparaissant dans moins de 5 documents
        max_df=0.8  # Ignore les mots trop fréquents
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Shape des features : {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# ==================== 5. ENTRAÎNEMENT DES MODÈLES ====================
def train_models(X_train, y_train, X_test, y_test):
    """Entraîne plusieurs modèles et compare leurs performances"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entraînement : {name}")
        print(f"{'='*50}")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))
    
    return results

# ==================== 6. VISUALISATION DES RÉSULTATS ====================
def visualize_results(results, y_test):
    """Visualise les performances des modèles"""
    
    # Comparaison des modèles
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Graphique des accuracies
    models_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in models_names]
    f1_scores = [results[name]['f1_score'] for name in models_names]
    
    axes[0, 0].bar(models_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy par modèle')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(models_names, f1_scores, color='coral')
    axes[0, 1].set_title('F1-Score par modèle')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Matrices de confusion pour les deux meilleurs modèles
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Matrice de confusion - {best_model_name}')
    axes[1, 0].set_ylabel('Vraie classe')
    axes[1, 0].set_xlabel('Classe prédite')
    
    # Comparaison finale
    comparison_df = pd.DataFrame({
        'Modèle': models_names,
        'Accuracy': accuracies,
        'F1-Score': f1_scores
    }).sort_values('Accuracy', ascending=False)
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=comparison_df.values,
                             colLabels=comparison_df.columns,
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Comparaison des modèles')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    print("📊 Graphique sauvegardé : model_comparison.png")
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"MEILLEUR MODÈLE : {best_model_name}")
    print(f"Accuracy : {results[best_model_name]['accuracy']:.4f}")
    print(f"F1-Score : {results[best_model_name]['f1_score']:.4f}")
    print(f"{'='*50}")
    
    return best_model_name

# ==================== 7. FONCTION DE PRÉDICTION ====================
def predict_sentiment(text, model, vectorizer, method='binary'):
    """Prédit le sentiment d'un nouveau texte"""
    # Preprocessing
    cleaned = preprocess_text(text)
    
    # Vectorisation
    text_tfidf = vectorizer.transform([cleaned])
    
    # Prédiction
    prediction = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    
    if method == 'binary':
        sentiment = "POSITIF" if prediction == 1 else "NÉGATIF"
        confidence = max(proba) * 100
    elif method == 'ternary':
        sentiments = {0: "NÉGATIF", 1: "NEUTRE", 2: "POSITIF"}
        sentiment = sentiments[prediction]
        confidence = max(proba) * 100
    else:  # multiclass
        sentiment = f"Score {prediction + 1}/5"
        confidence = max(proba) * 100
    
    return sentiment, confidence

# ==================== 8. SAUVEGARDE ET CHARGEMENT ====================
def get_best_existing_model(filepath='models', method='ternary'):
    """Récupère les informations du meilleur modèle existant"""
    import os
    import glob
    
    if not os.path.exists(filepath):
        return None, 0.0
    
    # Chercher tous les fichiers de métadonnées pour la méthode spécifiée
    metadata_files = glob.glob(f"{filepath}/metadata_{method}_*.json")
    
    if not metadata_files:
        return None, 0.0
    
    best_accuracy = 0.0
    best_metadata_path = None
    
    for metadata_path in metadata_files:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if metadata['accuracy'] > best_accuracy:
                    best_accuracy = metadata['accuracy']
                    best_metadata_path = metadata_path
        except:
            continue
    
    return best_metadata_path, best_accuracy

def save_model(model, vectorizer, method, results, filepath='models'):
    """Sauvegarde le modèle SEULEMENT s'il est meilleur que les précédents"""
    import os
    
    # Créer le dossier models s'il n'existe pas
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Vérifier si un meilleur modèle existe déjà
    best_metadata_path, best_existing_accuracy = get_best_existing_model(filepath, method)
    current_accuracy = results['accuracy']
    
    print(f"\n{'='*60}")
    print(f"📊 COMPARAISON DES PERFORMANCES")
    print(f"{'='*60}")
    print(f"🆕 Nouveau modèle - Accuracy : {current_accuracy:.4f}")
    
    if best_metadata_path:
        print(f"🏆 Meilleur modèle existant - Accuracy : {best_existing_accuracy:.4f}")
        
        if current_accuracy <= best_existing_accuracy:
            print(f"\n❌ Le nouveau modèle n'est pas meilleur. Sauvegarde annulée.")
            print(f"   Différence : {(best_existing_accuracy - current_accuracy):.4f}")
            print(f"{'='*60}\n")
            return None, None, None
        else:
            print(f"\n✅ Le nouveau modèle est MEILLEUR ! Sauvegarde en cours...")
            print(f"   Amélioration : +{(current_accuracy - best_existing_accuracy):.4f}")
    else:
        print(f"📁 Aucun modèle existant. Premier modèle sauvegardé.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder le modèle
    model_path = f"{filepath}/best_model_{method}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarder le vectorizer
    vectorizer_path = f"{filepath}/best_vectorizer_{method}.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Sauvegarder les métadonnées avec timestamp
    metadata = {
        'method': method,
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'accuracy': results['accuracy'],
        'f1_score': results['f1_score'],
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_path = f"{filepath}/metadata_{method}_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Sauvegarder aussi les métadonnées du meilleur modèle actuel
    best_metadata_path = f"{filepath}/best_metadata_{method}.json"
    with open(best_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ MODÈLE SAUVEGARDÉ AVEC SUCCÈS")
    print(f"{'='*60}")
    print(f"📁 Modèle : {model_path}")
    print(f"📁 Vectorizer : {vectorizer_path}")
    print(f"📁 Métadonnées : {metadata_path}")
    print(f"🏆 Type de modèle : {type(model).__name__}")
    print(f"📈 Accuracy : {results['accuracy']:.4f}")
    print(f"📈 F1-Score : {results['f1_score']:.4f}")
    print(f"{'='*60}\n")
    
    return model_path, vectorizer_path, metadata_path

def load_model(filepath='models', method='ternary'):
    """Charge le meilleur modèle sauvegardé"""
    import os
    
    model_path = f"{filepath}/best_model_{method}.pkl"
    vectorizer_path = f"{filepath}/best_vectorizer_{method}.pkl"
    metadata_path = f"{filepath}/best_metadata_{method}.json"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"❌ Aucun modèle trouvé pour la méthode '{method}'")
        return None, None
    
    print(f"📂 Chargement du meilleur modèle ({method})...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Charger les métadonnées si disponibles
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Modèle chargé : {metadata['model_type']}")
        print(f"📈 Accuracy : {metadata['accuracy']:.4f}")
        print(f"📈 F1-Score : {metadata['f1_score']:.4f}")
        print(f"📅 Date d'entraînement : {metadata.get('training_date', 'N/A')}")
    else:
        print("✅ Modèle chargé avec succès!")
    
    return model, vectorizer

# ==================== 9. PIPELINE PRINCIPAL ====================
def main(filepath, method='binary', sample_size=None):
    """
    Pipeline complet d'analyse sentimentale
    
    Parameters:
    - filepath: chemin vers le fichier CSV
    - method: 'binary' (positif/négatif), 'ternary' (positif/neutre/négatif), ou 'multiclass' (5 classes)
    - sample_size: nombre d'échantillons (None pour tout le dataset)
    """
    
    # Chargement
    df = load_data(filepath)
    
    # Exploration
    explore_data(df)
    
    # Préparation
    df_processed = prepare_data(df, method=method, sample_size=sample_size)
    
    # Split train/test
    X = df_processed['cleaned_text']
    y = df_processed['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTaille du training set : {len(X_train)}")
    print(f"Taille du test set : {len(X_test)}")
    
    # Création des features
    X_train_tfidf, X_test_tfidf, vectorizer = create_features(X_train, X_test)
    
    # Entraînement
    results = train_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Visualisation
    best_model_name = visualize_results(results, y_test)
    
    # Retour du meilleur modèle et du vectorizer
    best_model = results[best_model_name]['model']
    best_results = {
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score']
    }
    
    # Sauvegarde automatique du meilleur modèle
    save_model(best_model, vectorizer, method, best_results)
    
    return best_model, vectorizer, results

# ==================== EXEMPLE D'UTILISATION ====================
if __name__ == "__main__":
    # Chemin vers votre fichier CSV
    FILEPATH = "data/Reviews.csv"
    
    print("="*60)
    print("DÉMARRAGE DE L'ANALYSE SENTIMENTALE")
    print("="*60)
    
    # Vérification que le fichier existe
    import os
    if not os.path.exists(FILEPATH):
        print(f"❌ ERREUR : Le fichier {FILEPATH} n'existe pas!")
        print(f"📁 Dossier actuel : {os.getcwd()}")
        print(f"📄 Fichiers disponibles : {os.listdir('.')}")
        exit(1)
    
    print(f"✅ Fichier trouvé : {FILEPATH}\n")
    
    # Exécution du pipeline avec classification TERNAIRE (Positif/Neutre/Négatif)
    best_model, vectorizer, all_results = main(
        FILEPATH, 
        method='ternary',  # Changé de 'binary' à 'ternary' pour détecter les neutres
        sample_size=50000
    )
    
    # Test de prédiction sur de nouveaux textes
    print("\n" + "="*50)
    print("TEST DE PRÉDICTION")
    print("="*50)
    
    test_reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Don't waste your money.",
        "It's okay, nothing special but does the job.",
        "Not bad, meets expectations.",
        "Outstanding product! Highly recommended!"
    ]
    
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, best_model, vectorizer, method='ternary')
        print(f"\nAvis : {review}")
        print(f"Sentiment : {sentiment} (Confiance: {confidence:.2f}%)")