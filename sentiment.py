from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import argparse
import os
import json

app = FastAPI()

# Pydantic model for API input
class SentimentRequest(BaseModel):
    texts: Optional[List[str]] = None
    urls: Optional[List[str]] = None
    x_post_ids: Optional[List[str]] = None
    include_embeddings: bool = False
    desired_sentiment: Optional[str] = None

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com"
)

def scrape_content(url):
    """Scrape text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
        return text[:2000]  # Limit for efficiency
    except:
        return ""

def fetch_x_post(post_id):
    """Fetch X post content (placeholder, requires X API access)."""
    # Note: X API access is restricted. Use your existing X post analysis logic.
    try:
        # Placeholder: Replace with your X post fetching logic
        response = requests.get(f"https://api.x.com/2/tweets/{post_id}", headers={"Authorization": f"Bearer {os.getenv('X_API_KEY')}"})
        response.raise_for_status()
        return response.json().get("text", "")
    except:
        return ""

def analyze_sentiment(text):
    """Analyze sentiment using DeepSeek."""
    prompt = f"Analyze sentiment of: {text[:500]}. Return positive, negative, or neutral and a confidence score (0-1)."
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=20
        )
        result = resp.choices[0].message.content.strip()
        sentiment, confidence = result.split(" ") if " " in result else (result, "0.5")
        return sentiment, float(confidence)
    except:
        return "neutral", 0.5

def generate_recommendation(sentiment, text, desired_sentiment=None):
    """Generate actionable recommendation."""
    if desired_sentiment and sentiment.lower() != desired_sentiment.lower():
        return f"Address {sentiment.lower()} sentiment to align with {desired_sentiment.lower()}: {text[:50]}..."
    if sentiment.lower() == "negative":
        return f"Address negative sentiment in: {text[:50]}..."
    elif sentiment.lower() == "positive":
        return f"Reinforce positive sentiment in: {text[:50]}..."
    return "Maintain neutral sentiment"

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings using sentence-transformers."""
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])[0]
    return embedding

def summarize_sentiment(sentiments, texts):
    """Summarize sentiment distribution and themes."""
    dist = {"positive": 0, "negative": 0, "neutral": 0}
    for s in sentiments:
        dist[s.lower()] += 1
    total = sum(dist.values())
    dist = {k: f"{(v/total*100):.1f}%" if total else "0%" for k, v in dist.items()}

    # Summarize top themes for negative sentiment
    negative_texts = [t for t, s in zip(texts, sentiments) if s.lower() == "negative"]
    themes = "No negative sentiment found"
    if negative_texts:
        prompt = f"Summarize negative themes in: {', '.join(negative_texts[:5])}. Return a 3-5 word summary."
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=10
            )
            themes = resp.choices[0].message.content.strip()
        except:
            themes = "Negative sentiment detected"
    
    return {"distribution": dist, "negative_themes": themes}

@app.post("/sentiment-analysis")
def sentiment_analysis(request: SentimentRequest):
    try:
        # Collect texts
        texts, sources, urls = [], [], []
        if request.texts:
            texts.extend(request.texts)
            sources.extend(["manual"] * len(request.texts))
            urls.extend([""] * len(request.texts))
        if request.urls:
            for url in request.urls:
                text = scrape_content(url)
                if text:
                    texts.append(text)
                    sources.append("website")
                    urls.append(url)
        if request.x_post_ids:
            for post_id in request.x_post_ids:
                text = fetch_x_post(post_id)
                if text:
                    texts.append(text)
                    sources.append("X")
                    urls.append(f"https://x.com/post/{post_id}")
        
        if not texts:
            raise ValueError("No valid text inputs provided")

        # Analyze sentiment
        results = []
        sentiments = []
        for text, source, url in zip(texts, sources, urls):
            sentiment, confidence = analyze_sentiment(text)
            recommendation = generate_recommendation(sentiment, text, request.desired_sentiment)
            result = {
                "text": text[:500],
                "source": source,
                "url": url,
                "sentiment": sentiment,
                "confidence": confidence,
                "recommendation": recommendation
            }
            if request.include_embeddings:
                result["embedding"] = ",".join(map(str, extract_keywords_and_embeddings(text)))
            results.append(result)
            sentiments.append(sentiment)

        # Summarize sentiment
        summary = summarize_sentiment(sentiments, texts)

        # Save to CSV
        df = pd.DataFrame(results)
        output_cols = ["text", "source", "url", "sentiment", "confidence", "recommendation"]
        if request.include_embeddings:
            output_cols.append("embedding")
        df[output_cols].to_csv("sentiment_output.csv", index=False)

        # Generate bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(summary["distribution"].keys()),
                y=[float(v.strip("%")) for v in summary["distribution"].values()],
                marker_color=["#2ca02c", "#d62728", "#7f7f7f"]
            )
        ])
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Percentage (%)",
            width=600, height=400
        )
        fig.write_html("sentiment_visualization.html")

        return {
            "results": results,
            "summary": summary,
            "csv_output": "sentiment_output.csv",
            "visualization": "sentiment_visualization.html"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main(input_csv=None, output_csv="sentiment_output.csv", output_html="sentiment_visualization.html", include_embeddings=False, desired_sentiment=None):
    """Run sentiment analysis from command line."""
    if input_csv:
        df = pd.read_csv(input_csv)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain 'text' column")
        texts = df["text"].tolist()
        sources = df.get("source", ["manual"] * len(texts)).tolist()
        urls = df.get("url", [""] * len(texts)).tolist()
    else:
        texts, sources, urls = [], [], []
    
    results = []
    sentiments = []
    for text, source, url in zip(texts, sources, urls):
        sentiment, confidence = analyze_sentiment(text)
        recommendation = generate_recommendation(sentiment, text, desired_sentiment)
        result = {
            "text": text[:500],
            "source": source,
            "url": url,
            "sentiment": sentiment,
            "confidence": confidence,
            "recommendation": recommendation
        }
        if include_embeddings:
            result["embedding"] = ",".join(map(str, extract_keywords_and_embeddings(text)))
        results.append(result)
        sentiments.append(sentiment)

    summary = summarize_sentiment(sentiments, texts)
    df = pd.DataFrame(results)
    output_cols = ["text", "source", "url", "sentiment", "confidence", "recommendation"]
    if include_embeddings:
        output_cols.append("embedding")
    df[output_cols].to_csv(output_csv, index=False)

    fig = go.Figure(data=[
        go.Bar(
            x=list(summary["distribution"].keys()),
            y=[float(v.strip("%")) for v in summary["distribution"].values()],
            marker_color=["#2ca02c", "#d62728", "#7f7f7f"]
        )
    ])
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Percentage (%)",
        width=600, height=400
    )
    fig.write_html(output_html)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved visualization: {output_html}")
    return {"results": results, "summary": summary}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Sentiment Analysis Add-On")
    parser.add_argument("--input", help="Input CSV with text, source, url columns")
    parser.add_argument("--output-csv", default="sentiment_output.csv", help="Output CSV path")
    parser.add_argument("--output-html", default="sentiment_visualization.html", help="Output HTML path")
    parser.add_argument("--embeddings", action="store_true", help="Include embeddings for clustering")
    parser.add_argument("--desired-sentiment", help="Desired sentiment (positive, negative, neutral)")
    args = parser.parse_args()
    main(args.input, args.output_csv, args.output_html, args.embeddings, args.desired_sentiment)
