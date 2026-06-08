import re
import numpy as np

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def summarize_text(text):

    # Remove citations
    text = re.sub(r'\[[^\]]*\]', '', text)

    sentences = sent_tokenize(text)

    if len(sentences) <= 3:
        return text

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = np.asarray(
        tfidf_matrix.sum(axis=1)
    ).flatten()

    # Give importance to first and last sentences
    scores[0] *= 1.5
    scores[-1] *= 1.3

    ranked_indices = scores.argsort()[::-1]

    # Select top 40% sentences
    num_sentences = max(
        2,
        int(len(sentences) * 0.4)
    )

    selected_indices = sorted(
        ranked_indices[:num_sentences]
    )

    summary = " ".join(
        [sentences[i] for i in selected_indices]
    )

    return summary