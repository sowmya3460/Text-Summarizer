from flask import Flask, render_template, request

from utils.summarizer import summarize_text
from utils.fuzzy_score import similarity_score
from utils.preprocess import clean_text

import nltk

# Download required NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""
    score = 0

    original_count = 0
    summary_count = 0

    if request.method == "POST":

        paragraph = request.form.get("paragraph", "")

        if paragraph.strip():

            # Clean input text
            cleaned_text = clean_text(paragraph)

            # Generate summary
            summary = summarize_text(cleaned_text)

            # Calculate fuzzy similarity score
            score = similarity_score(
                cleaned_text,
                summary
            )

            # Word counts
            original_count = len(
                cleaned_text.split()
            )

            summary_count = len(
                summary.split()
            )

            # Debug (check terminal output)
            print("Original Words:", original_count)
            print("Summary Words:", summary_count)
            print("Accuracy:", score)

    return render_template(
        "index.html",
        summary=summary,
        score=score,
        original_count=original_count,
        summary_count=summary_count
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )