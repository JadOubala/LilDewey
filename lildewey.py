import urllib.request
import fitz  # PyMuPDF (takes PDF)
import re
from tqdm import tqdm
from translate import Translator
from langdetect import detect

# Designed by: Jad Oubala, Will Kaminski & Sam Bradley

# Function to download PDF from a user-inputted URL
# Uses the urllib.request.urlretrieve function to fetch the PDF from the url and
# save it as a file at output_path.

# def download_pdf(url, output_path):
    # urllib.request.urlretrieve(url, output_path)

# Preprocessing function to clean text
# Cleans up the extracted text by removing newlines and extra spaces.

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

# Convert PDF document to text
# Opens the PDF using fitz.open (PyMuPDF).
# Iterates through the specified page range, extracting text from each page.
# Applies the preprocess function to clean up each page's text.
# Collects and returns a list of the cleaned text strings, one for each page.

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None or end_page > total_pages:
        end_page = total_pages

    text_list = []

    for i in tqdm(range(start_page-1, end_page), desc="Extracting text from PDF"):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

# Convert list of texts to smaller chunks
# Iterates through the list of preprocessed text strings (texts).
# For each text string, splits it into words and then groups into chunks
# If the end of a chunk falls short of the word_length and it's not the last
# chunk, the remaining words are prepended to the next text string to avoid
# having short ending chunk.
# Each chunk is prefixed with its page number and enclosed in quotes.

def text_to_chunks(texts, word_length=150, start_page=1):
    chunks = []
    buffer = []

    for idx, text in enumerate(texts):
        words = text.split(' ')
        for word in words:
            buffer.append(word)
            if len(buffer) >= word_length:
                chunk = ' '.join(buffer).strip()
                chunks.append(f'Page {idx+start_page}: "{chunk}"')
                buffer = []

        # Handle the remaining buffer if it's long enough
        if len(buffer) >= word_length:
            chunk = ' '.join(buffer).strip()
            chunks.append(f'Page {idx+start_page}: "{chunk}"')
            buffer = []

    return chunks

# Example usage input file:
file_path = '/content/GANPaper.pdf'  # If directly uploaded
# output_path = 'downloaded_document.pdf'
# download_pdf(url, output_path)

texts = pdf_to_text(file_path, start_page=1)
chunks = text_to_chunks(texts, word_length=150)

# Optionally, print or process the chunks
        # for chunk in chunks[:5]:  # Print first 5 chunks as a sample
        #     print(chunk)

# Chunk Embedding:

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming `chunks` is your list of preprocessed text chunks
embeddings = model.encode(chunks, show_progress_bar=True)

import faiss
import numpy as np

dimension = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
index.add(embeddings.astype(np.float32))  # Add embeddings to index


# Querying the Index for Relevant Chunks
# create function to query the index with a user's question
# find the most relevant chunks, and display them:
# Also, account for translational logic!

def search(query, k=5):

    original_language = detect_lang(query)
    query_in_english = translate_to_english(query) if original_language != 'en' else query

    query_embedding = model.encode([query_in_english])[0].astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), 5)

    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks, original_language


# Example query (TESTING)
    # query = "Explain the concept of interaction in markets between local producers and local consumers."
    # results = search(query)

    # for result in results:
    #     print(result)


# TRANSLATION-ADJACENT FUNCTIONS

# Translates text safely by splitting it into segments, translating each segment,
# and then concatenating the results.

def safe_translate(text, from_lang, to_lang, max_length=500):
    translator = Translator(to_lang=to_lang, from_lang=from_lang)
    # Split text into segments of max_length characters without breaking words
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:  # +1 for space
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_length = len(word)
        else:
            current_segment.append(word)
            current_length += len(word) + 1  # +1 for space

    # Add the last segment if it's not empty
    if current_segment:
        segments.append(" ".join(current_segment))

    # Translate each segment
    translated_segments = [translator.translate(segment) for segment in segments]

    # Combine translated segments
    translated_text = " ".join(translated_segments)
    return translated_text

def detect_lang(text):
    return detect(text)

def translate_to_english(text, max_length=500):
    detected_language = detect_lang(text)
    if detected_language != 'en':
        return safe_translate(text, from_lang=detected_language, to_lang='en', max_length=max_length)
    return text

def translate_from_english(text, target_lang, max_length=500):
    if target_lang != 'en':
        return safe_translate(text, from_lang='en', to_lang=target_lang, max_length=max_length)
    return text



# GPT Integration:
from openai import OpenAI

client = OpenAI(api_key='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')


# Takes the semantically searched chunks as input and generates a response using
# the ChatGPT API
def generate_response_from_chunks(user_query, max_tokens=325):
    relevant_chunks, original_language = search(user_query)

    if original_language != 'en':
        translated_query_to_english = translate_to_english(user_query)
    else:
        translated_query_to_english = user_query  # Use the original query directly if it's already in English


    # Start constructing the prompt with more structured guidance
    prompt = "search results:\n\n" + "".join([f"{i+1}. {chunk}\n\n" for i, chunk in enumerate(relevant_chunks)])
    prompt += "Instructions: Compose a comprehensive and succinct reply to the query using the search results given. " \
              "Cite each reference using [Page #number] notation (every result has a number at the beginning). " \
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects " \
              "with the same name, create separate answers for each. Only include information found in the results and " \
              "don't add any additional information. Make sure the answer is correct and don't output false content. You should also mention where a given answer might be found in the text if appropriate. Keep answers under around seven sentences." \
              "If the text does not relate to the query, simply state 'Sorry, Lil' Dewey found nothing relevant in the text.'. Don't write 'Answer:' " \
              "Directly start and state the answer.\n"

    prompt += f"Query: {translated_query_to_english}\n\n"

    # Send the prompt to the ChatGPT model using the chat/completions endpoint
    response = client.chat.completions.create(model="gpt-4",  # Specify the chat model you're using
                                               messages=[
                                                   {"role": "system", "content": prompt},
                                                   {"role": "user", "content": "Please provide a response based on the above instructions."}
                                               ],
                                               temperature=0.7,
                                               max_tokens=max_tokens,
                                               top_p=1.0,
                                               frequency_penalty=0.0,
                                               presence_penalty=0.0)

    # Extracting and returning the text from the response:
    generated_text = response.choices[0].message.content.strip()

    translated_response = translate_from_english(generated_text, original_language) if original_language != 'en' else generated_text

    return translated_response

# Example usage
# Example user query in a non-English language
user_query = "¿Cuál es el papel que desempeñan las redes generativas adversarias en términos de mejorar la eficacia de los modelos de IA?"
response = generate_response_from_chunks(user_query)
print("Response:", response)
