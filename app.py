from flask import Flask, request, jsonify
import requests
import fitz  # PyMuPDF for PDF processing
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

os.makedirs('uploads', exist_ok=True)

# Load environment variables
subscription_key = os.getenv('SUBSCRIPTION_KEY')
region = os.getenv('REGION')

# Azure Text Analytics credentials
text_analytics_endpoint = os.getenv('TEXT_ANALYTICS_ENDPOINT')
text_analytics_key = os.getenv('TEXT_ANALYTICS_KEY')

# Azure OpenAI credentials
openai_api_base = os.getenv('OPENAI_API_BASE')
openai_api_version = os.getenv('OPENAI_API_VERSION')
deployment_name = os.getenv('DEPLOYMENT_NAME')
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_api_type = "azure"

# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    openai_api_base=openai_api_base,
    openai_api_version=openai_api_version,
    deployment_name=deployment_name,
    openai_api_key=openai_api_key,
    openai_api_type=openai_api_type,
)

# Define the prompt template for summarization
template = """
You are a highly skilled research analyst. Your task is to analyze and summarize an earnings call transcript for a company. Read the full transcript text of the earnings call and analyze the main topics discussed. Identify the key points and significant highlights based on the content of the transcript. Create a detailed summary or notes that capture the essence of the call, focusing on the most important aspects of the discussion. Review the transcript and extract the main topics discussed. Your summary should be concise, with clear and structured bullet points under relevant headings. Use appropriate headings based on the content. Each bullet point should highlight a key takeaway from the call.
{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

# Helper function to authenticate Azure Text Analytics client
def authenticate_client():
    ta_credential = AzureKeyCredential(text_analytics_key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=text_analytics_endpoint, 
        credential=ta_credential)
    return text_analytics_client

# Function to convert MP3 to text using Azure Speech-to-Text
def convert_audio_to_text(audio_file_path):
    endpoint = f'https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-05-15-preview'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Accept': 'application/json'
    }
    files = {
        'audio': open(audio_file_path, 'rb'),
        'definition': ('', '{"locales":["en-US"], "profanityFilterMode": "Masked", "channels": [0,1]}')
    }
    response = requests.post(endpoint, headers=headers, files=files)
    if response.status_code == 200:
        response_data = response.json()
        combined_phrases = response_data.get("combinedPhrases", [])
        full_text = " ".join(phrase["text"] for phrase in combined_phrases)
        return full_text
    else:
        raise Exception(f"Error in speech-to-text conversion: {response.status_code}, {response.json()}")

# Function to extract text from PDF and perform NER
def process_pdf_for_ner(pdf_path):
    client = authenticate_client()
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document:
        text += page.get_text()
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    all_filtered_entities = []
    for batch in batch_documents(chunks):
        response_entities = recognize_entities(client, batch)
        filtered_entities = filter_entities(response_entities)
        all_filtered_entities.extend(filtered_entities)
    return all_filtered_entities

# Helper functions for NER
def batch_documents(documents, batch_size=5):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def recognize_entities(client, documents):
    response_entities = client.recognize_entities(documents=documents)
    return response_entities

def filter_entities(response_entities):
    categories = ['Person', 'PersonType', 'Organization', 'Location', 'Event', 'Product', 'Skill']
    filtered_entities = []
    for entity_list in response_entities:
        filtered_entities.extend([
            entity for entity in entity_list.entities
            if entity.confidence_score > 0.9 and entity.category in categories
        ])
    return filtered_entities

# Function to perform fuzzy matching and correct transcript
def context_aware_fuzzy_match_replace(transcript, entities, phrases, threshold=65):
    corrected_transcript = transcript
    entity_categories = ['Person', 'PersonType', 'Organization', 'Location', 'Event', 'Product', 'Skill']
    lower_phrases = [phrase.lower() for phrase in phrases]
    for entity in entities:
        if entity.category in entity_categories:
            entity_text_lower = entity.text.lower()
            match, score = process.extractOne(entity_text_lower, lower_phrases, scorer=fuzz.ratio)
            if score >= threshold:
                original_match = next((phrase for phrase in phrases if phrase.lower() == match), match)
                corrected_transcript = corrected_transcript.replace(entity.text, original_match, 1)
    return corrected_transcript

# Function to extract text from a PDF file for summarization
def extract_text_from_pdf(pdf_stream):
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/process', methods=['POST'])
def process_files():
    try:
        audio_file = request.files.get('audio')
        pdf_file = request.files.get('pdf')

        if not audio_file or not pdf_file:
            return jsonify({'error': 'Both audio and PDF files are required'}), 400

        audio_file_path = os.path.join('uploads', audio_file.filename)
        pdf_file_path = os.path.join('uploads', pdf_file.filename)

        audio_file.save(audio_file_path)
        pdf_file.save(pdf_file_path)

        # Convert audio to text
        transcript = convert_audio_to_text(audio_file_path)

        # Perform NER on PDF
        previous_entities = process_pdf_for_ner(pdf_file_path)

        # Perform NER on transcript in chunks
        client = authenticate_client()
        transcript_chunks = [transcript[i:i+5000] for i in range(0, len(transcript), 5000)]
        transcript_entities = []
        for chunk in transcript_chunks:
            response_entities = recognize_entities(client, [chunk])
            for entity_list in response_entities:
                transcript_entities.extend(filter_entities([entity_list]))

        # Extract phrases from previous entities
        phrases = [entity.text for entity in previous_entities]

        # Correct transcript using NER results and phrases from previous PDF
        corrected_transcript = context_aware_fuzzy_match_replace(transcript, transcript_entities, phrases)

        # Clean up uploaded files
        os.remove(audio_file_path)
        os.remove(pdf_file_path)

        return jsonify({'corrected_text': corrected_transcript})

    except Exception as e:
        app.logger.error(f'Error: {str(e)}')  # Log the error
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400

    input_text = request.form['text']  # Get the text from the request

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"text": input_text})

    return jsonify({'summary': result})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

