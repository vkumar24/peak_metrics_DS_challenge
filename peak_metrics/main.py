from data_processor import S3DataProcessor
from clean_string import CleanString
from utility import *
from ner_trainer import NerTrainer
from sentiment import SentimentTrainer
import re
import pandas as pd
import json
import spacy
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore")
import configparser
from keywords import *
config = configparser.ConfigParser()
config.read('credential.ini')  # Provide the path to your configuration file

# Accessing configuration values
aws_access_key_id = config.get('AWS', 'aws_access_key_id')
aws_secret_access_key = config.get('AWS', 'aws_secret_access_key')
bucket_name = config.get('S3', 'bucket_name')
directory_names = config.get('S3', 'directory_names').split(',')

class Main:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, directory_names, weight_audience=0.7,
                 weight_sentiment=0.3):
        self.data_processor = S3DataProcessor(aws_access_key_id, aws_secret_access_key, bucket_name, directory_names)
        self.clean_string = CleanString()
        self.sentiment_trainer = SentimentTrainer()
        self.ner_trainer = NerTrainer()
        self.weight_audience = weight_audience
        self.weight_sentiment = weight_sentiment

    def __call__(self):
        dfs_social, dfs_news, dfs_blog = self.data_processor.process_data()
        social_df = pd.concat(dfs_social)
        news_df = pd.concat(dfs_news)
        social_df['title'] = social_df.apply(lambda x: self.clean_string(x.title), axis=1)
        social_df['social_stats'] = social_df.apply(lambda x: json.loads(x.social_stats.decode('utf-8')), axis=1)
        social_df['audience_visits'] = social_df.social_stats.apply(lambda x: x.get('audience_visits', 0))
        social_df['global_rank'] = social_df.social_stats.apply(lambda x: x.get('global_rank', 0))
        news_df['source'] = news_df['source'].fillna('')
        news_df['summary'] = news_df['summary'].fillna('')
        news_df['headline'] = news_df['headline'].fillna('')
        news_df['journalist'] = news_df['journalist'].fillna('')
        news_df['body'] = news_df['body'].fillna('')
        news_df['news_stats'] = news_df.apply(lambda x: json.loads(x.news_stats.decode('utf-8')), axis=1)
        news_df['category'] = news_df.apply(lambda x: x.news_stats.get('category', ''), axis=1)
        news_df['global_rank'] = news_df.apply(lambda x: x.news_stats.get('global_rank', ''), axis=1)
        news_df['audience_visits'] = news_df.apply(lambda x: x.news_stats.get('audience_visits', 0.0), axis=1)
        news_df['audience_visits'] = news_df['audience_visits'].astype(int)
        news_df['alexa_rank'] = news_df.apply(lambda x: x.news_stats.get('alexa_rank', ''), axis=1)
        news_df['newsguard'] = news_df.apply(lambda x: x.news_stats.get('newsguard', {}), axis=1)
        news_df['newsguard_score'] = news_df.apply(lambda x: x.newsguard.get('score', 0.0), axis=1)
        news_df['newsguard_orientation'] = news_df.apply(lambda x: x.newsguard.get('orientation', ''), axis=1)

        # compute sentiment score and assign label
        social_df['title_sentiment_score_label'] = social_df.apply(lambda x: self.sentiment_trainer(x.title), axis=1)
        social_df['title_sentiment'] = social_df.apply(lambda x: x.title_sentiment_score_label[0], axis=1)
        social_df['sentiment_label'] = social_df.apply(lambda x: x.title_sentiment_score_label[1], axis=1)
        # compute sentiment score and assign label
        news_df['headline_sentiment_score_label'] = news_df.apply(lambda x: self.sentiment_trainer(x.headline), axis=1)
        news_df['headline_sentiment'] = news_df.apply(lambda x: x.headline_sentiment_score_label[0], axis=1)
        news_df['sentiment_label'] = news_df.apply(lambda x: x.headline_sentiment_score_label[1], axis=1)
        # Weight assigned to title sentiment score
        max_audience_visits = max(social_df.audience_visits)
        social_df['normalized_audience_visits'] = social_df.apply(lambda x: x.audience_visits / max_audience_visits,
                                                                  axis=1)
        social_df['post_importance'] = social_df.apply(
            lambda x: calculate_post_importance(x.normalized_audience_visits, x.title_sentiment, self.weight_audience,
                                                self.weight_sentiment), axis=1)

        news_max_audience_visits = max(news_df.audience_visits)
        news_df['normalized_audience_visits'] = news_df.apply(lambda x: x.audience_visits / news_max_audience_visits,
                                                              axis=1)
        news_df['post_importance'] = news_df.apply(
            lambda x: calculate_post_importance(x.normalized_audience_visits, x.headline_sentiment, self.weight_audience,
                                                self.weight_sentiment), axis=1)

        # finding airline related titles
        social_df['is_title_airline'] = social_df.apply(lambda x: is_airline_related_text(x.title), axis=1)
        # filtering airline related titles
        social_airline_df = social_df[social_df.is_title_airline == True]
        # finding airline related titles
        news_df['is_headline_airline'] = news_df.apply(lambda x: is_airline_related_text(x.headline), axis=1)
        # filtering airline related titles
        news_airline_df = news_df[news_df.is_headline_airline == True]
        # computing airline brands from title and headline
        social_airline_df['airline_brands'] = social_airline_df.apply(lambda x: self.ner_trainer(x.title), axis=1)
        news_airline_df['airline_brands'] = news_airline_df.apply(lambda x: self.ner_trainer(x.headline), axis=1)
        # check united_airlines posts
        social_airline_df['is_united_airlines'] = social_airline_df.apply(
            lambda x: check_airline_brand(x.airline_brands, 'united'), axis=1)
        # check southwest_airlines posts
        social_airline_df['is_southwest_airlines'] = social_airline_df.apply(
            lambda x: check_airline_brand(x.airline_brands, 'southwest'), axis=1)
        # check united_airlines posts
        news_airline_df['is_united_airlines'] = news_airline_df.apply(
            lambda x: check_airline_brand(x.airline_brands, 'united'), axis=1)
        # check southwest_airlines posts
        news_airline_df['is_southwest_airlines'] = news_airline_df.apply(
            lambda x: check_airline_brand(x.airline_brands, 'southwest'), axis=1)

        # Filtering united_airlines dataframe
        is_social_united_airlines_df = social_airline_df[social_airline_df.is_united_airlines == True]
        # Filtering southwest_airlines dataframe
        is_social_southwest_airlines_df = social_airline_df[social_airline_df.is_southwest_airlines == True]
        # Filtering united_airlines dataframe for Negative sentiments
        is_social_united_airlines_neg_sent_df = is_social_united_airlines_df[
            is_social_united_airlines_df.sentiment_label == 'Negative']
        # Sorting united_airlines negative sentiments dataframe by post_importance(more imp. if post_importance greater)
        is_social_united_airlines_neg_sent_df = is_social_united_airlines_neg_sent_df.sort_values('post_importance',
                                                                                                  ascending=True)
        # Filtering southwest_airlines dataframe for Negative sentiments
        is_social_southwest_airlines_neg_sent_df = is_social_southwest_airlines_df[
            is_social_southwest_airlines_df.sentiment_label == 'Negative']
        is_social_southwest_airlines_neg_sent_df = is_social_southwest_airlines_neg_sent_df.sort_values(
            'post_importance', ascending=True)
        social_united_airlines_neg_titles = dict(
            zip(is_social_united_airlines_neg_sent_df.post_importance, is_social_united_airlines_neg_sent_df.title))

        # Outputting united airlines viral negative sentiment posts
        with open("output/united_airlines_viral_negative_posts.json", "w") as f:
            json.dump(social_united_airlines_neg_titles, f)

        # Printing united airlines viral negative sentiment posts and corresponding peak importance
        for title_, post_importance_ in zip(is_social_united_airlines_neg_sent_df.title[:100],
                                            is_social_united_airlines_neg_sent_df.post_importance[:100]):
            print('post_titile:')
            print(title_)
            print('post_importance = ', post_importance_)
            print("\n")
        southwestair_neg_titles = dict(zip(is_social_southwest_airlines_neg_sent_df.post_importance,
                                           is_social_southwest_airlines_neg_sent_df.title))

        # Outputting southwest airlines viral negative sentiment posts
        with open("output/southwestair_viral_negative_posts.json", "w") as f:
            json.dump(southwestair_neg_titles, f)

        # Printing southwest airlines viral negative sentiment posts and corresponding peak importance
        for title_ in is_social_southwest_airlines_neg_sent_df.title:
            print(title_)
            print("\n")
        social_airline_df_ = social_airline_df[
            ['title', 'is_united_airlines', 'is_southwest_airlines', 'sentiment_label']]
        news_airline_df_ = news_airline_df[
            ['headline', 'is_united_airlines', 'is_southwest_airlines', 'sentiment_label']]

        social_airline_df_['title/headline'] = social_airline_df_['title']
        news_airline_df_['title/headline'] = news_airline_df_['headline']
        # Concatenate the dataframes vertically
        digital_airline_df = pd.concat([social_airline_df_, news_airline_df_], ignore_index=True)
        digital_is_united_airlines_df = digital_airline_df[digital_airline_df.is_united_airlines == True]
        digital_is_southwest_airlines_df = digital_airline_df[digital_airline_df.is_southwest_airlines == True]
        # topic modelling with lda, extracting topics, keywords and weights
        airline_industry_topics = get_topics(digital_airline_df, 'title/headline', 5, 10)
        # plotting topics and keywords distribution
        get_topic_modelling_plot(airline_industry_topics)
        # topic modelling with lda, extracting topics, keywords and weights dataframe
        airline_industry_topics_df = get_topic_keyword_weights(airline_industry_topics)
        airline_industry_topics_df.to_csv("output/airline_industry_topics.csv", index=False)
        # topic modelling with lda, extracting topics, keywords and weights
        united_air_topics = get_topics(digital_is_united_airlines_df, 'title/headline', 5, 10)
        # plotting topics and keywords distribution
        get_topic_modelling_plot(united_air_topics)
        # topic modelling with lda, extracting topics, keywords and weights dataframe
        united_air_topics_df = get_topic_keyword_weights(united_air_topics)
        united_air_topics_df.to_csv("output/united_air_topics.csv", index=False)
        # topic modelling with lda, extracting topics, keywords and weights
        digital_is_united_airlines_neg_sent_df = digital_is_united_airlines_df[
            digital_is_united_airlines_df.sentiment_label == 'Negative']
        united_air_negative_topics = get_topics(digital_is_united_airlines_neg_sent_df, 'title/headline', 5, 10)
        # plotting topics and keywords distribution
        get_topic_modelling_plot(united_air_negative_topics)
        # topic modelling with lda, extracting topics, keywords and weights dataframe
        united_air_negative_topics_df = get_topic_keyword_weights(united_air_negative_topics)
        united_air_negative_topics_df.to_csv("output/united_air_negative_topics.csv", index=False)
        # topic modelling with lda, extracting topics, keywords and weights
        southwest_air_topics = get_topics(digital_is_southwest_airlines_df, 'title/headline', 5, 10)
        # plotting topics and keywords distribution
        get_topic_modelling_plot(southwest_air_topics)
        # topic modelling with lda, extracting topics, keywords and weights dataframe
        southwest_air_topics_df = get_topic_keyword_weights(southwest_air_topics)
        southwest_air_topics_df.to_csv("output/southwest_air_topics.csv", index=False)
        # topic modelling with lda, extracting topics, keywords and weights
        digital_is_southwest_airlines_neg_sent_df = digital_is_southwest_airlines_df[
            digital_is_southwest_airlines_df.sentiment_label == 'Negative']
        southwest_air_negative_topics = get_topics(digital_is_southwest_airlines_neg_sent_df, 'title/headline', 5, 10)
        # plotting topics and keywords distribution
        get_topic_modelling_plot(southwest_air_negative_topics)
        # topic modelling with lda, extracting topics, keywords and weights dataframe
        southwest_air_negative_topics_df = get_topic_keyword_weights(southwest_air_negative_topics)
        southwest_air_negative_topics_df.to_csv("output/southwest_air_negative_topics.csv", index=False)
        other_airline_df = digital_airline_df[
            (digital_airline_df.is_united_airlines == False) & (digital_airline_df.is_southwest_airlines == False)]
        other_airline_df_neg_sent_df = other_airline_df[other_airline_df.sentiment_label == 'Negative']
        other_airline_df_topics = get_topics(other_airline_df, 'title/headline', 5, 10)
        get_topic_modelling_plot(other_airline_df_topics)
        other_airline_df_topics_df = get_topic_keyword_weights(other_airline_df_topics)
        other_airline_df_topics_df.to_csv("output/other_airline_df_topics.csv", index=False)
        # Preprocessing the titles in the DataFrame
        social_airline_df['preprocessed_text'] = social_airline_df['title'].apply(preprocess_text)

        # Creating the TF-IDF matrix for reference posts
        reference_posts = ['united wife', 'united anthonybass']
        # reference_posts = ['southwest delai', 'southwest worst']
        preprocessed_reference_posts = [preprocess_text(post) for post in reference_posts]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_reference_posts)

        # Computing similarity scores for each post
        social_airline_df['similarity_score'] = social_airline_df['preprocessed_text'].apply(
            lambda x: get_max_similarity(x, tfidf_matrix))
        social_airline_df['len_airline_brands'] = social_airline_df.apply(lambda x: len(x.airline_brands), axis=1)
        df_airlines_no_brand = social_airline_df[social_airline_df.len_airline_brands == 0]

        united_related_df = df_airlines_no_brand.query('similarity_score<1.0')
        united_related_df = united_related_df.sort_values('similarity_score', ascending=False)
        united_related_sent_negative_df = united_related_df[united_related_df.sentiment_label == 'Negative']
        # Create a regex pattern to match airline mentions
        pattern = r"\b({})\b".format("|".join(re.escape(airline) for airline in AIRLINE_KEYWORDS), flags=re.IGNORECASE)
        united_related_sent_negative_df['title'] = united_related_sent_negative_df.apply(lambda x: x.title.lower(),
                                                                                         axis=1)
        # Filter the DataFrame based on airline mention
        non_explicit_united_air_df = united_related_sent_negative_df[
            ~united_related_sent_negative_df['title'].str.contains(pattern, regex=True)]
        # non_explicit_southwest_air_df = sw_related_sent_negative_df[~sw_related_sent_negative_df['title'].str.contains(pattern, regex=True)]
        for i in non_explicit_united_air_df.title:
            print(i)
        non_explicit_united_air_df_ = non_explicit_united_air_df[['title', 'similarity_score']]
        non_explicit_united_air_df_.to_csv("output/non_explicit_united_airline_posts.csv", index=False)


# Initilize and run Main class
main = Main(aws_access_key_id, aws_secret_access_key, bucket_name, directory_names, 0.7, 0.3)
print(main())