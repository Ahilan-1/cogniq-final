import aiohttp
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from markupsafe import Markup
from functools import lru_cache
from retrying import retry
import re
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

MAX_RESULTS = 10
MIN_SUMMARIES = 5

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="!secret")

templates = Jinja2Templates(directory="templates")

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ERROR_PHRASES = [
    "Access Denied", "No useful summary available", "Your access to the NCBI website",
    "possible misuse/abuse situation", "has been temporarily blocked", "is not an indication of a security issue",
    "a run away script", "to restore access", "please have your system administrator contact",
    "Log In", "Continue with phone number", "Email or username", "you agree to our Terms of Service", "Cookie Policy, Privacy Policy and Content Policies." "Password", "Forgot password","cookies","Accept","Cookie Settings"
]

@lru_cache(maxsize=128)
async def google_search(query, session):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with session.get(url, headers=headers) as response:
        return await response.text()

def parse_search_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for g in soup.find_all('div', class_='tF2Cxc')[:MAX_RESULTS]:
        title = g.find('h3').text if g.find('h3') else 'No title found'
        link = g.find('a')['href']
        snippet = g.find('div', class_='VwiC3b').text if g.find('div', class_='VwiC3b') else 'No snippet found'
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

def extract_youtube_video_id(url):
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    youtube_match = re.match(youtube_regex, url)
    if youtube_match and 'channel' not in url:
        return youtube_match.group(6)
    return None

@retry(stop_max_attempt_number=3, wait_fixed=1000)
async def fetch_article(url, session):
    async with session.get(url, timeout=10) as response:
        if response.status == 200:
            return await response.text()
        else:
            response.raise_for_status()

def is_valid_summary(summary):
    return not any(phrase in summary for phrase in ERROR_PHRASES)

def clean_summary(summary):
    """
    This function removes any unwanted phrases from the summary after it has been generated.
    """
    for phrase in ERROR_PHRASES:
        summary = summary.replace(phrase, "")
    return summary.strip()

async def fetch_and_summarize(url, session):
    try:
        video_id = extract_youtube_video_id(url)
        if video_id:
            # Handle YouTube transcript fetching
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summary = clean_summary(summary)  # Clean the summary before returning it
            if is_valid_summary(summary):
                return summary
        else:
            # Handle general article summarization
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            summary = clean_summary(article.summary)  # Clean the summary before returning it
            if is_valid_summary(summary):
                return summary
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        try:
            html = await fetch_article(url, session)
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs[:5]])
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summary = clean_summary(summary)  # Clean the summary before returning it
            if is_valid_summary(summary):
                return summary
        except Exception as e:
            logger.error(f"Error fetching summary for {url}: {e}")
            return None

def create_bullet_point_summary(text):
    sentences = sent_tokenize(text)
    bullet_points = '\n'.join([f"• {sentence}" for sentence in sentences])
    return bullet_points

def combine_summaries(summaries):
    combined_text = " ".join(summaries)
    if combined_text:
        return create_bullet_point_summary(combined_text)
    return "No useful summary available."

def filter_results(results):
    return [result for result in results if not any(phrase in result['snippet'] for phrase in ERROR_PHRASES)]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    async with aiohttp.ClientSession() as session:
        google_html = await google_search(query, session)
        google_results = parse_search_results(google_html)
        google_results = filter_results(google_results)

        google_summaries = await asyncio.gather(*[fetch_and_summarize(result['link'], session) for result in google_results])
        google_summaries = [summary for summary in google_summaries if summary]

        if len(google_summaries) >= MIN_SUMMARIES:
            google_combined_summary = combine_summaries(google_summaries)
        else:
            google_combined_summary = "No useful summary available."

        return templates.TemplateResponse("result.html", {
            "request": request,
            "query": query,
            "google_results": google_results,
            "google_combined_summary": Markup(google_combined_summary),
        })

import json

@app.get("/suggestions", response_class=JSONResponse)
async def get_suggestions(query: str):
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={query}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()  # Read the response as plain text
                try:
                    data = json.loads(text)  # Parse the text as JSON
                    return data[1]  # Return the suggestions list
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse suggestions for query: {query}")
                    return []
    return []


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
