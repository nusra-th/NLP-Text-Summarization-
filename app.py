from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load summarization model (CPU, PyTorch)
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    framework="pt",
    device=-1
)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    article = ""

    if request.method == "POST":
        article = request.form.get("article", "")

        if len(article.strip()) > 50:
            result = summarizer(
                article,
                max_length=130,
                min_length=40,
                do_sample=False
            )
            summary = result[0]["summary_text"]
        else:
            summary = "Please enter a longer article."

    return render_template("index.html", summary=summary, article=article)

if __name__ == "__main__":
    app.run(debug=True)

