# AI-Powered Text Summarizer & Keyword Extractor | LLM + NLP Project  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/) [![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)](https://www.nltk.org/)  

> A **deep learning + NLP project** that uses **LLMs (BART/DistilBART)** for **text summarization**, combined with **TF-IDF & POS tagging** for keyword extraction and topic modeling.  
> This project demonstrates how to build **AI-powered applications** with **real-world NLP pipelines**. 

---

## Project Overview  

This project is a compact **AI-based Python application** that accepts user text input and produces meaningful insights:  
- **Text Summarization** → Using pre-trained **LLMs** (BART / DistilBART).  
- **Keyword Extraction** → With **TF-IDF** and **POS tagging**.  
- **Topic Detection** → Based on extracted keywords.  
- **Text Statistics** → Word count, sentence count, character count.  

It showcases the integration of **transformer-based models** with classical NLP, making it suitable for both **research and production-grade applications**.  

---

## Features

- Summarizes text into **short, medium, long** mode.  
- Extracts top **keywords & topics** using hybrid techniques.  
- Provides detailed **text statistics**. 

---

## Requirements

- **Python 3.10+** 
- Libraries:
  ```bash
  pip install transformers torch scikit-learn nltk
  ```
- NLTK data is downloaded automatically by the code.

---

## Usage

```python
from text_summarizer import TextSummarizer, display_results

# Initialize the analyzer
analyzer = TextSummarizer()

# Sample text input
sample_text = "Your text here..."

# Analyze text (options: 'short', 'medium', 'long')
result = analyzer.analyze_text(sample_text, summary_length='medium')

# Display results
display_results(result)
```

---

## Input Guidelines

- **Minimum input length:** 50 words  
- **Maximum input length:** No strict limit (memory-dependent for very large texts)  
- **Language supported:** English

---

## Output Format

The output includes:
```json
{
  "original_text_stats": {"word_count": 0, "sentence_count": 0, "character_count": 0},
  "summary": {"summary": "...", "compression_ratio": 0.0, "original_length": 0, "summary_length": 0},
  "keywords": ["kw1", "kw2", "..."],
  "topics": ["topic1", "topic2", "..."],
  "analysis_timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

---

## Example Output

```
Words: 120 | Sentences: 4 | Characters: 620

Summary:
Artificial intelligence allows machines to perform human-like tasks, including reasoning, problem-solving, and decision-making.

Keywords: artificial intelligence, machines, tasks, reasoning, decision-making

Topics: artificial intelligence, machines, tasks, reasoning, decision-making

Completed at: 2025-09-16 21:00:00
```

## Why This Project?

This isn’t just a summarizer — it’s a demonstration of how **modern LLMs** can be combined with **classical NLP techniques** to create **real-world, AI-powered applications**.

If you’re looking for:

- **LLM projects**
- **Deep learning web apps**
- **NLP pipelines**
- **Practical AI/ML applications**

This repo is a great starting point.

---

## Author

Hi, I’m **Saad Umar**  

 **BS in Information Technology**  
Passionate about **Machine Learning, Deep Learning, and NLP**  
Building **AI-powered applications** to solve **real-world challenges**

---

## Let’s Connect

[GitHub](https://github.com/Saadumar26)  
[LinkedIn](https://www.linkedin.com/in/muhammad-saad-umar-632a4a28a/)  
[X (Twitter)](https://x.com/SaadUmar26)  

---

## Support

If this project helped you:

- **Star this repo**  
- **Fork & contribute**  
- **Follow my journey → [GitHub Profile](https://github.com/Saadumar26)**



## License

MIT License

