import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import numpy as np
import matplotlib.pyplot as plt


# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Read the Moby Dick file with 'utf-8' encoding
with open('mobydick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
tokens = word_tokenize(text)

# Filter out punctuation tokens
filtered_tokens = [token for token in tokens if token.lower() not in string.punctuation]

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in filtered_tokens if token.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for (word, tag) in pos_tags)
most_common_pos = pos_counts.most_common(5)

# In the part marked by POS, add the count of each part of speech to the part-of-speech counter:
for word, tag in pos_tags:
    if tag in pos_counts:
        pos_counts[tag] += 1
    else:
        pos_counts[tag] = 1

print("Most common parts of speech:")
for pos, count in most_common_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_lemmas = []
for token, pos in pos_tags[:20]:
    if pos.startswith('N'):
        lemma = lemmatizer.lemmatize(token, pos='n')  # Lemmatize nouns
    elif pos.startswith('V'):
        lemma = lemmatizer.lemmatize(token, pos='v')  # Lemmatize verbs
    else:
        lemma = lemmatizer.lemmatize(token)  # Default to noun lemmatization
    top_lemmas.append(lemma)

print("\nTop 20 lemmas:")
for lemma in top_lemmas:
    print(lemma)

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = [sia.polarity_scores(token)['compound'] for token in tokens]
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

print(f"\nAverage sentiment score: {average_sentiment}")

if average_sentiment > 0.05:
    overall_sentiment = "Positive"
elif average_sentiment < 0.05:
    overall_sentiment = "Negative"
else:
    overall_sentiment = "Neutral"

print(f"Overall text sentiment: {overall_sentiment}")

# Create a list of parts of speech and frequencies to be displayed according to the part-of-speech counter.
pos = list(pos_counts.keys())
freq = list(pos_counts.values())

# Create a function to draw a histogram.
def plot_bar_chart(x, y, xlabel, ylabel, title):
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Call the function of drawing histogram and pass in the list of parts of speech and frequency.
plot_bar_chart(pos, freq, 'Parts of Speech', 'Frequency', 'POS Frequency')