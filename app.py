from flask import Flask, render_template, request
from summarizer import TextSummarizer  # Import your class from Python file

app = Flask(__name__)
analyzer = TextSummarizer()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_text = request.form.get("input_text", "")
        summary_length = request.form.get("summary_length", "medium")
        if user_text.strip():
            result = analyzer.analyze_text(user_text, summary_length=summary_length)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
