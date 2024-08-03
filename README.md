# Cogniq Search Engine

**Cogniq Search Engine** is an advanced web application designed to streamline the search experience by integrating Google and YouTube searches, providing concise summaries, and delivering real-time query suggestions. It aims to help users quickly find relevant information by summarizing web content and videos.

## Key Features

1. **Integrated Search**:
   - Perform simultaneous searches on Google and YouTube.
   - Display search results from both platforms in a unified interface.

2. **Content Summarization**:
   - Automatically summarize articles and YouTube video transcripts.
   - Highlight key sentences and important information for quick understanding.

3. **Real-Time Suggestions**:
   - Offer real-time search query suggestions as users type.
   - Enhance user experience with relevant search terms.

4. **Error Handling and Filtering**:
   - Filter out irrelevant or low-quality search results.
   - Ensure the provided summaries are useful and informative.

## Technologies Used

- **FastAPI**: For building a high-performance web API.
- **aiohttp**: For asynchronous HTTP requests.
- **BeautifulSoup4**: For parsing HTML and XML content.
- **Newspaper3k**: For extracting and summarizing articles.
- **YouTube Transcript API**: For fetching transcripts of YouTube videos.
- **Transformers**: For natural language processing and summarization.
- **NLTK**: For text processing and analysis.
- **Jinja2**: For rendering HTML templates.
- **Uvicorn**: For running the ASGI server.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Ahilan-1/cogniq-final.git
    cd cogniq-final
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1. Run the application:

    ```python
    python app.py
    ```

2. Open your browser and navigate to `http://127.0.0.1:8000` to access Cogniq Search Engine.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue on GitHub.

---

Cogniq Search Engine combines the power of Google and YouTube searches with advanced summarization techniques to enhance the search experience. It delivers concise and relevant information, helping users find what they need quickly and efficiently.
