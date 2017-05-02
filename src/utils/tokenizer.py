from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


def tokenize(text):
    """NLTK's treebank tokenizer + lemmatizing. Works on more raw text."""
    tokens = TreebankWordTokenizer().tokenize(text)
    tokens = lemmatize(tokens)
    tokens = filter(lambda s: len(s) > 2, tokens)  # remove tokens with < 3 chars
    return tokens


def lemmatize(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    return map(wordnet_lemmatizer.lemmatize, tokens)


def stem(tokens):
    stemmer = SnowballStemmer("english")
    return map(stemmer.stem, tokens)
