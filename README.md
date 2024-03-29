# LilDewey
Forgot a detail in a huge textbook or need concise answers from complex papers? LilDewey, your chatbot assistant, quickly finds info in an English text PDF that you provide, and can respond to and answer questions in over 183 languages!

Designed by Jad Oubala

Language functionality by Will Kaminski

Testing by Sam Bradley

**HOW TO USE:**
In order to interact with the model, please engage with the following instructions:

1. First run the dependencies which contains the necessary to install libraries.
2. (Depending on interface used, download a PDF and copy its file path).
3. Proceed to **line 80** and fill the filepath in (ensure it is within single quotes): file_path = '____'
4. After this, proceed to **line 231**, where you may ask any question (in english or any other language) and fill it in between the double quotes. user_query = "____"
5. Then run the kernal involving the main code, and see Lil' Dewey's response to your prompt (and it's detected language)!

**What it does:**
Anyone in secondary education knows the struggle of massive textbooks and compounding this issue is the difficulty experienced by students who may use english as a second language. DeweyBot is an embedded model that specializes in the capabilities offered by the OpenAI API and GPT-4 to be quickly trained on any given corpus of text and then answer specific user queries in coherent, concise and content-specific responses that cite page numbers and sections accordingly. Furthermore, in recognition of the difficulty many students who may speak english as a second language face, DeweyBot is able to leverage an internal translational model that can take in queries in over 183 languages, and then produce responses in that same detected language while still based on the original given english text. This could be invaluable for, say, a spanish-speaking student learning a biology textbook in English, but feels she could best formulate the question in her native tongue in order to understand a given concept.

**How it was built:**
Lil’ Dewey is a comprehensive solution designed for processing PDF documents to extract text, segmenting the text into manageable chunks, embedding these chunks for semantic similarity searches, and optionally translating content for multilingual support. The solution leverages several Python libraries and integrates with the OpenAI API for generating responses based on the content of the PDF. Here's an overview of how each part of the code works and integrates into the broader solution:

PDF Download: Utilizes urllib.request.urlretrieve to download a PDF from a specified URL to a local file. This allows users to input PDFs via URLs for processing. Text Extraction from PDF: Employs fitz (PyMuPDF) to open PDF documents and extract text. This process is enhanced by a preprocessing function that cleans the text by removing newlines and excess spaces, improving the quality of text for further processing. Text Chunking: After cleaning, the text is split into smaller, manageable chunks. This is crucial for analyzing large documents, as it breaks down the text into segments that are easier to process and understand in context.

After the data pre-processing we began the semantic search processing and search chunk embedding: We used SentenceTransformer to convert text chunks into dense vector embeddings, enabling semantic search capabilities. These embeddings capture the contextual meaning of text chunks, facilitating the identification of relevant content based on query semantics. FAISS Indexing: Integrates with FAISS (Facebook AI Similarity Search) to create an efficient similarity search index from the embeddings (and utilizing the K-Nearest Neighbors (KNN) Algorithm). This index supports fast retrieval of text chunks that are semantically close to a given query. Semantic Search: We then implemented a search function that encodes a user query into an embedding and uses the FAISS index to find the most relevant text chunks. This function supports queries in different languages by detecting the query's language and translating it to English if necessary. This is the first dimension of the translational capability.

For this we leveraged ‘langdetect’ for language detection and a Translator for translating text. This supports multilingual queries by translating non-English queries into English before processing and then translating responses back into the original query language for maximal accessibility and comprehension for the user. This is the second dimension of the translational capability. The model accommodates detecting and translating for over 183 languages.

Next we built integration with OpenAI for Response Generation. OpenAI API Interaction: We incorporated OpenAI's API to generate responses based on the semantically relevant text chunks. It constructs a prompt that includes the relevant chunks and instructions for generating a coherent and contextually appropriate response. This allows leveraging advanced language models (like GPT) for synthesizing information from the PDF into a concise answer. It also includes page numbers, sections, and relevant information that directly cites the text to minimize hallucination or misattribution of information.
