import aiohttp
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
from bs4 import BeautifulSoup
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
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
nltk.download('stopwords', quiet=True)

MAX_RESULTS = 10
MIN_SUMMARIES = 5

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="!secret")

templates = Jinja2Templates(directory="templates")

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ERROR_PHRASES = [
    "Access Denied",
    "No useful summary available",
    "Your access to the NCBI website",
    "possible misuse/abuse situation",
    "has been temporarily blocked",
    "is not an indication of a security issue",
    "a run away script",
    "to restore access",
    "please have your system administrator contact",
    "Log In",
    "Continue with phone number",
    "Email or username",
    "Password",
    "Forgot password"
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
        title = g.find('h3')
        title = title.text if title else 'No title found'
        link = g.find('a')['href']
        snippet = g.find('div', class_='VwiC3b')
        snippet = snippet.text if snippet else 'No snippet found'
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

def extract_youtube_video_id(url):
    video_id = None
    youtube_regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_match = re.match(youtube_regex, url)
    if youtube_match and 'channel' not in url:
        video_id = youtube_match.group(6)
    return video_id

@retry(stop_max_attempt_number=3, wait_fixed=1000)
async def fetch_article(url, session):
    async with session.get(url, timeout=10) as response:
        if response.status == 200:
            return await response.text()
        else:
            response.raise_for_status()

def is_valid_summary(summary):
    for phrase in ERROR_PHRASES:
        if phrase in summary:
            return False
    return True

async def fetch_and_summarize(url, session):
    try:
        video_id = extract_youtube_video_id(url)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = ' '.join([entry['text'] for entry in transcript])
            summary = ' '.join(sent_tokenize(text)[:3])
            if is_valid_summary(summary):
                return summary
        else:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            if is_valid_summary(article.summary):
                return article.summary
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        try:
            html = await fetch_article(url, session)
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs[:5]])
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            if is_valid_summary(summary):
                return summary
        except Exception as e:
            logger.error(f"Error fetching summary for {url}: {e}")
            return None

@retry(stop_max_attempt_number=3, wait_fixed=1000)
async def fetch_youtube_search(query, session):
    url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    async with session.get(url, timeout=10) as response:
        return await response.text()

def parse_youtube_search(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for video in soup.find_all('a', href=True):
        if '/watch' in video['href']:
            title = video.get('title')
            if not title:
                title = video.find('img')['alt'] if video.find('img') else 'No title found'
            link = f"https://www.youtube.com{video['href']}"
            results.append({'title': title, 'link': link})
    logger.debug(f"Parsed YouTube search results: {results}")
    return results[:MAX_RESULTS]

def highlight_important_sentences(text, query, num_sentences=3):
    sentences = sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    freq_dist = FreqDist(words)
    query_words = set(query.lower().split())
    
    def sentence_importance(sentence):
        sentence_words = set(word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum())
        return sum(freq_dist[word] for word in sentence_words) + sum(5 for word in sentence_words if word in query_words)
    
    ranked_sentences = sorted([(sentence, sentence_importance(sentence)) for sentence in sentences], 
                              key=lambda x: x[1], reverse=True)
    
    highlighted_sentences = set(sentence for sentence, _ in ranked_sentences[:num_sentences])
    
    highlighted_text = []
    for sentence in sentences:
        if sentence in highlighted_sentences:
            highlighted_text.append(f'<mark class="highlight">{sentence}</mark>')
        else:
            highlighted_text.append(sentence)
    
    return ' '.join(highlighted_text)

def combine_summaries(summaries, query):
    combined_text = " ".join(summaries)
    if combined_text:
        sentences = sent_tokenize(combined_text)
        summary = ' '.join(sentences[:6])
        return highlight_important_sentences(summary, query)
    return "No useful summary available."

def filter_results(results):
    filtered_results = []
    for result in results:
        if any(phrase in result['snippet'] for phrase in ERROR_PHRASES):
            continue
        filtered_results.append(result)
    return filtered_results

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    async with aiohttp.ClientSession() as session:
        google_html = await google_search(query, session)
        google_results = parse_search_results(google_html)
        google_results = filter_results(google_results)

        youtube_html = await fetch_youtube_search(query, session)
        youtube_results = parse_youtube_search(youtube_html)
        
        google_summaries = []
        youtube_summaries = []

        google_tasks = [fetch_and_summarize(result['link'], session) for result in google_results]
        youtube_tasks = [fetch_and_summarize(result['link'], session) for result in youtube_results]
        
        google_summaries_results = await asyncio.gather(*google_tasks)
        youtube_summaries_results = await asyncio.gather(*youtube_tasks)

        for summary, result in zip(google_summaries_results, google_results):
            if summary and is_valid_summary(summary):
                highlighted_summary = highlight_important_sentences(summary, query)
                result['summary'] = Markup(highlighted_summary)
                google_summaries.append(summary)

        for summary, result in zip(youtube_summaries_results, youtube_results):
            if summary and is_valid_summary(summary):
                highlighted_summary = highlight_important_sentences(summary, query)
                result['summary'] = Markup(highlighted_summary)
                youtube_summaries.append(summary)

        if len(google_summaries) >= MIN_SUMMARIES:
            google_combined_summary = combine_summaries(google_summaries, query)
        else:
            google_combined_summary = "No useful summary available."

        if len(youtube_summaries) >= MIN_SUMMARIES:
            youtube_combined_summary = combine_summaries(youtube_summaries, query)
        else:
            youtube_combined_summary = "No useful summary available."

        return templates.TemplateResponse("index.html", {
            "request": request,
            "query": query,
            "google_results": google_results,
            "youtube_results": youtube_results,
            "google_combined_summary": Markup(google_combined_summary),
            "youtube_combined_summary": Markup(youtube_combined_summary)
        })

@app.get("/suggestions", response_class=JSONResponse)
async def get_suggestions(query: str):
    suggestions = []
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={query}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                suggestions = data[1]
    return suggestions

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
