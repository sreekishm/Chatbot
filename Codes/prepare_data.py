from replace_words import replace_words
from extract_keywords import extract_keywords
import spacy
nlp = spacy.load("en_core_web_sm")
def prepare_data(text):
    text=replace_words(text)
    keywords=extract_keywords(text)
    sentences=[]
    doc=nlp(text)
    for sent in doc.sents:
        sentences.append(sent.text)
    keyword_sentences = {}

    for keyword in keywords:
        keyword_sentences[keyword] = []
        for sentence_number, sentence in enumerate(sentences, start=0):
            if keyword.lower() in sentence.lower():
                keyword_sentences[keyword].append(sentence_number)
    return keyword_sentences, sentences