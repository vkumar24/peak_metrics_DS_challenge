import re
import pandas as pd
import warnings
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
from keywords import *

vectorizer = TfidfVectorizer()

# Function to compute the post importance
def calculate_post_importance(normalized_audience_visits, title_sentiment_score, weight_audience, weight_sentiment):
    # Normalizing the audience visits
    # Computing the post importance
    post_importance = (weight_audience * normalized_audience_visits) + (weight_sentiment * title_sentiment_score)
    return post_importance


def check_airline_brand(texts, airline_brand):
    keyword = airline_brand.lower()
    for text in texts:
        if keyword in text.lower():
            return True
    return False


# Function to find airline related posts
def is_airline_related_text(text):
    # Constructing the regex pattern
    regex_pattern = r"\b(" + "|".join(map(re.escape, AIRLINE_KEYWORDS)) + r")\b"
    # Compiling the regex pattern
    regex = re.compile(regex_pattern, re.IGNORECASE)
    matches = regex.findall(text)
    if matches:
        return True
    return False


def get_topics(df, col, num_topics, num_words):
    df['processed_str'] = df.apply(lambda x: preprocess_string(x[col]), axis=1)
    dictionary = corpora.Dictionary(df['processed_str'])
    corpus = [dictionary.doc2bow(str_) for str_ in df['processed_str']]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.show_topics(num_topics, num_words)
    coherence_model = CoherenceModel(model=lda_model, texts=df['processed_str'], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print("Coherence Score:", coherence_score)
    return topics


# Function to extract hashtags
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags


# Function to transform topics, keywords and weights into dataframe
def get_topic_keyword_weights(topics_list):
    topics = list()
    for topics_list_ in topics_list:
        topic_list = [
            (float(value.split('*')[0].strip()), value.split('*')[1].replace('"', '').strip())
            for value in topics_list_[1].split(' + ')]
        topics.append(topic_list)

    data = {"TopicID": [], "Keywords": [], "Weights": []}
    for idx, topic in enumerate(topics):
        weights, keywords = zip(*topic)
        data["TopicID"].append(idx)
        data["Keywords"].append(", ".join(keywords))
        data["Weights"].append(", ".join([str(weight) for weight in weights]))
        topics_df = pd.DataFrame(data)

    return topics_df


# Function to plot Topic Modelling result
def get_topic_modelling_plot(topics_list, plot_title):
    topics = list()
    for topics_list_ in topics_list:
        topic_list = [
            (float(value.split('*')[0].strip()), value.split('*')[1].replace('"', '').strip())
            for value in topics_list_[1].split(' + ')]
        topics.append(topic_list)

    # Finding the maximum number of keywords
    max_keywords = max(len(topic) for topic in topics)

    # Padding the topic weights with zeros for topics with fewer keywords
    padded_topics = [topic + [(0, '')] * (max_keywords - len(topic)) for topic in topics]

    # Extracting topic labels and weights
    topic_labels = [", ".join([word for _, word in topic]) for topic in padded_topics]
    topic_weights = np.array([[weight for weight, _ in topic] for topic in padded_topics])

    # Creating bar plots for topic weights
    plt.figure(figsize=(15, 10))
    x = np.arange(len(topic_labels))

    for i in range(max_keywords):
        plt.bar(x + i * 0.15, topic_weights[:, i], width=0.05, alpha=0.9, label=f"Keyword {i + 1}")

    plt.xlabel("Topics")
    plt.ylabel("Weights")
    plt.xticks(x, topic_labels, rotation=45, ha="right")
    plt.title(f"Topic Modeling Results - {plot_title} Topic Keywords Weights")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Function to preprocess the text
def preprocess_text(text):
    preprocessed_text = text.lower()
    preprocessed_text = re.sub('[^a-zA-Z0-9\s]', '', preprocessed_text)
    return preprocessed_text


# Function to calculate the maximum similarity score between a post and reference posts
def get_max_similarity(vectorizer, preprocessed_text, tfidf_matrix):
    post_vector = vectorizer.transform([preprocessed_text])
    similarity_scores = cosine_similarity(post_vector, tfidf_matrix)
    max_similarity = similarity_scores.max()
    return max_similarity