import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
def extract_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in words
        if word.lower() not in stop_words
        and word not in string.punctuation
    ]
    tagged_words = nltk.pos_tag(words)
    keywords = set()
    for word, tag in tagged_words:
        if tag.startswith("NN"):
            keywords.add(word)
        elif tag.startswith("VB"):
            keywords.add(word)
        elif tag.startswith("JJ"):
            keywords.add(word)

    return keywords