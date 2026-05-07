
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import joblib


class LDATopicModel:
    def __init__(self, num_topics=10, max_features=5000, random_state=42):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=2,
            max_features=max_features
        )
        self.model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=random_state
        )

    def fit_transform(self, documents):
        X = self.vectorizer.fit_transform(documents)
        topic_probs = self.model.fit_transform(X)
        return topic_probs.argmax(axis=1)

    def transform(self, documents):
        X = self.vectorizer.transform(documents)
        topic_probs = self.model.transform(X)
        return topic_probs.argmax(axis=1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)