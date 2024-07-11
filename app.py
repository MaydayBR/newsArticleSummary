import os
import requests
from bs4 import BeautifulSoup
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key'

# Function to scrape the latest news articles
def scrape_latest_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Assuming the website has <article> tags for news articles
    articles = soup.find_all('article')
    news_snippets = []
    for article in articles:
        description = article.find('p')  # Assuming each article has a <p> tag with description
        if description:
            news_snippets.append(description.get_text())
    return news_snippets

# Define a template for the prompt
template = """
Here are some recent news snippets:
{news_snippets}

Based on the above information, please answer the following question:
{user_question}
"""

# Function to create the RAG-based response
def generate_response(user_question, news_url):
    # Scrape the latest news from the provided URL
    news_snippets = scrape_latest_news(news_url)
    combined_snippets = "\n".join(news_snippets[:5])  # Use the first 5 news snippets

    # Prepare the prompt
    prompt = template.format(news_snippets=combined_snippets, user_question=user_question)

    # Initialize OpenAI LLM
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Create a LangChain LLMChain
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt))

    # Generate the response
    response = llm_chain.predict()

    return response

# Define a route to handle the news scraping and response generation
@app.route('/generate-response', methods=['POST'])
def generate_response_route():
    data = request.json
    news_url = data.get('news_url')
    user_question = data.get('user_question')

    if not news_url or not user_question:
        return jsonify({'error': 'news_url and user_question are required'}), 400

    try:
        response = generate_response(user_question, news_url)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
