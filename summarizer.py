import re
from datetime import datetime
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class TextSummarizer:
    def __init__(self):
        """Initialize summarizer and keyword extractor."""
        print("Loading summarization model...")
        self.load_summarization_model()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        print("Model ready!")

    def load_summarization_model(self):
        """Load BART or fallback DistilBART summarizer (CPU)."""
        try:
            model_name = "facebook/bart-large-cnn"
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=-1  # Force CPU
            )
        except Exception as e:
            print(f"Error loading BART: {e}")
            print("Falling back to DistilBART...")
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )

    def clean_text(self, text):
        """Basic cleaning of input text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()

    def extract_keywords_tfidf(self, text, num_keywords=10):
        """Extract keywords using TF-IDF."""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            sentences = [text]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1
        keyword_scores = list(zip(feature_names, mean_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in keyword_scores[:num_keywords]]

    def extract_keywords_simple(self, text, num_keywords=10):
        """Backup keyword extractor using POS tagging + frequency."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        pos_tags = pos_tag(words)
        keywords = [w for w, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
        word_freq = Counter(keywords)
        return [w for w, _ in word_freq.most_common(num_keywords)]

    def summarize_text(self, text, max_length=150, min_length=50):
        """Summarize given text with dynamic length."""
        text = self.clean_text(text)
        word_count = len(text.split())

        # If text is too short, return as-is
        if word_count < 50:
            return {
                "summary": text,
                "compression_ratio": 1.0,
                "note": "Text too short to summarize meaningfully"
            }

        # Dynamically adjust max_length based on input
        if word_count < max_length:
            max_length = int(word_count * 0.8)
            if max_length < min_length:
                max_length = word_count

        summary_result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        summary = summary_result[0]['summary_text']
        summary_words = len(summary.split())
        compression_ratio = round(summary_words / word_count, 2)
        return {
            "summary": summary,
            "compression_ratio": compression_ratio,
            "original_length": word_count,
            "summary_length": summary_words
        }

    def extract_topics(self, text, num_topics=5):
        """Extract topics based on keywords."""
        keywords = self.extract_keywords_tfidf(text, num_keywords=20)
        return keywords[:num_topics] if keywords else self.extract_keywords_simple(text, num_topics)

    def analyze_text(self, text, summary_length="medium"):
        """Full pipeline: summarize + keywords + topics + stats."""
        length_params = {
            "short": {"max_length": 100, "min_length": 30},
            "medium": {"max_length": 200, "min_length": 50},
            "long": {"max_length": 300, "min_length": 100}
        }
        params = length_params.get(summary_length, length_params["medium"])
        summary_result = self.summarize_text(text, **params)
        keywords = self.extract_keywords_tfidf(text, num_keywords=10)
        topics = self.extract_topics(text, num_topics=5)
        stats = {
            "word_count": len(text.split()),
            "sentence_count": len(sent_tokenize(text)),
            "character_count": len(text)
        }
        return {
            "original_text_stats": stats,
            "summary": summary_result,
            "keywords": keywords,
            "topics": topics,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ---------- Display Function ----------
def display_results(result):
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print("\nTEXT ANALYSIS RESULTS")
    stats = result["original_text_stats"]
    print(f"Words: {stats['word_count']} | Sentences: {stats['sentence_count']} | Characters: {stats['character_count']}")
    print(f"\nSummary:\n{result['summary']['summary']}")
    print(f"\nKeywords: {', '.join(result['keywords'])}")
    print(f"Topics: {', '.join(result['topics'])}")
    print(f"\nCompleted at: {result['analysis_timestamp']}")


# ---------- Test ----------
if __name__ == "__main__":
    analyzer = TextSummarizer()
    sample_text = """
    Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. High-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., language models and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore.
    """
    result = analyzer.analyze_text(sample_text, summary_length="medium")
    display_results(result)

