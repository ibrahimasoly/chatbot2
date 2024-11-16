import nltk

# Téléchargement des ressources nécessaires
nltk.download('punkt')         # Pour la tokenisation des phrases
nltk.download('stopwords')         # Pour la liste des mots vides
nltk.download('wordnet')           # Pour la lemmatisation
nltk.download('omw-1.4')           # Pour les données supplémentaires de WordNet
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st


with open('bookv2.txt', 'r', encoding='utf-8') as f :

    data = f.read().replace('\n', ' ')

# Tokeniser le texte en phrases

sentences = sent_tokenize(data, language="french")

# Définir une fonction pour prétraiter chaque phrase

def preprocess(sentence) :

    # Tokenize the sentence into words (Tokenisation de la phrase en mots)

    words = word_tokenize(sentence)

    return words



# Prétraitement de chaque phrase du texte

corpus = [preprocess(sentence) for sentence in sentences]

# Construction d'une matrice TF-IDF pour les phrases
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([' '.join(sentence) for sentence in corpus])

def get_most_relevant_sentence_tfidf(query):
    # Transformer la requête en vecteur TF-IDF
    query_vec = vectorizer.transform([' '.join(preprocess(query))])

    # Calculer les similarités cosinus entre la requête et les phrases
    similarities = cosine_similarity(query_vec, tfidf_matrix)

    # Trouver l'indice de la phrase la plus pertinente
    idx = similarities.argmax()

    # Retourner la phrase correspondante
    return ' '.join(corpus[idx])

def chatbot(question) :

    # Trouver la phrase la plus pertinente

    most_relevant_sentence = get_most_relevant_sentence_tfidf(question)

    # Retourne la réponse

    return most_relevant_sentence
# Créer une application Streamlit



st.title("Chatbot")

st.write("Bonjour ! Je suis un chatbot. Demandez-moi n'importe quoi sur le sujet de recommandation musicale.")
st.write("Je suis là pour vous répondre. 😊")

    # Obtenir la question de l'utilisateur

question = st.text_input("Vous: ")

    # Créer un bouton pour soumettre la question

if st.button("Envoyer"):

        # Appeler la fonction chatbot avec la question et afficher la réponse

    response = chatbot(question)

        # Nettoyer la réponse
    words = response.split(' ; ')  # Séparer les mots
    sentence = ' '.join(words)  # Joindre les mots avec un espace

        # Capitaliser la première lettre et ajouter un point à la fin
    sentence = sentence.capitalize() 
    if response=="":
        st.write("Desoler j'ai pas de reponse concernant cette reponse")
    else:
        st.write("Chatbot : " + sentence)


