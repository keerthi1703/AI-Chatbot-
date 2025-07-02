from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import wikipedia
from newsapi import NewsApiClient
from serpapi import GoogleSearch
import time
import re
import os
from dotenv import load_dotenv

# Load environment variables from the .env file(secure api keys)
load_dotenv()

app = Flask(__name__)#we initialize the flask app here

#  pipeline = pre trained models QA - question answering model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=-1 
)

# fetch the api keys from .env file
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

def clean_answer(answer):
    """Clean and format the answer"""
    if not answer:
        return None
    # Remove special characters and extra whitespace
    answer = re.sub(r'[^\w\s-]', '', answer.strip())
    # Remove citation markers like [1], [2]
    answer = re.sub(r'\[\d+\]', '', answer)
    return answer[:500]  # Limit answer length

def get_best_source(query):
    """Dynamically select the best source"""
    if any(word in query.lower() for word in ["current", "recent", "today", "latest"]):
        return "newsapi"
    elif any(word in query.lower() for word in ["what is", "who was", "define"]):
        return "wikipedia"
    else:
        return "serpapi"

def get_serpapi_answer(query):
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,  
            "hl": "en"
        }
        result = GoogleSearch(params).get_dict()
        
        # Try knowledge graph
        if 'knowledge_graph' in result:
            kg = result['knowledge_graph']
            if 'title' in kg:
                return kg['title']
        
        # Try featured snippet
        snippet = result.get("organic_results", [{}])[0].get("snippet", "")
        if len(snippet.split()) < 30:
            return snippet
    except:
        return None
    
def get_wikipedia_answer(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None
    
def get_newsapi_answer(query):
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)  # Replace
        articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=1
        )
        if articles['articles']:
            return articles['articles'][0]['title']
    except:
        return None
    




def get_roberta_answer(question, context):
    """Use Roberta QA model on the given context"""
    try:
        result = qa_pipeline(question=question, context=context)
        return clean_answer(result['answer'])
    except Exception as e:
        print(f"Roberta QA error: {e}")
        return None

def get_answer(question):
    """Main function to get verified answers using Roberta when applicable"""
    if not question.strip():
        return "Please ask a question."

    source = get_best_source(question)
    answer = None

    if source == "newsapi":
        answer = get_newsapi_answer(question)

    elif source == "wikipedia":
        context = get_wikipedia_answer(question)
        if context:
            answer = get_roberta_answer(question, context)

    elif source == "serpapi":
        context = get_serpapi_answer(question)
        if context and len(context.split()) >= 30:
            answer = get_roberta_answer(question, context)
        else:
            answer = context  # use snippet directly if Roberta context too short

    # Fallback loop if primary source failed
    if not answer:
        for fallback_source in ["wikipedia", "serpapi", "newsapi"]:
            if fallback_source != source:
                if fallback_source == "newsapi":
                    answer = get_newsapi_answer(question)
                else:
                    context = globals()[f"get_{fallback_source}_answer"](question)
                    if context:
                        answer = get_roberta_answer(question, context) if fallback_source != "newsapi" else context
                if answer:
                    break

    return answer or "I couldn't find a reliable answer. Try rephrasing or providing more context."

@app.route("/") #loads the homepage first
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"answer": "Please enter a question."})
    
    start_time = time.time()
    try:
        answer = get_answer(question)
        elapsed = time.time() - start_time
        
        # Rate limiting to prevent API abuse
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
            
        return jsonify({
            "answer": answer,
            "processing_time": f"{elapsed:.2f}s"
        })
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({"answer": "An error occurred while processing your question."}), 500

if __name__ == "__main__":
    app.run(debug=True)