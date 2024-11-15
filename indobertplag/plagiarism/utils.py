import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load IndoBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('LazarusNLP/all-indobert-base-v4')
model = AutoModel.from_pretrained('LazarusNLP/all-indobert-base-v4')
model.eval()  # Set model to evaluation mode

def remove_headers_footers(page, margin=50):
    # Try to remove headers and footers
    blocks = page.get_text("blocks")
    page_height = page.rect.height
    filtered_text = ""

    for block in blocks:
        x0, y0, x1, y1, text = block[:5]
        if y0 > margin and y1 < (page_height - margin):
            filtered_text += text + "\n"

    return filtered_text

def clean_text(text):
    keywords = ['Abstrak', 'Pendahuluan', 'Kata kunci-?']
    key_pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b")
    journal_pattern = re.compile(
        r"computatio:? journal of computer science and information systems,? volume \d+,? no\.? \d+,? "
        r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember) \d{4}",
        re.IGNORECASE
    )
    final_pattern = re.compile(r"[^a-zA-Z0-9.,!?\s]+", re.IGNORECASE)
    text = re.sub(journal_pattern, "", text)  # Remove journal mentions
    text = re.sub(final_pattern, "", text)  # Remove special characters
    text = re.sub(key_pattern, '', text)  # Remove keywords
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1] [2]
    text = re.sub(r'\.+', '.', text)  # Replace multiple periods with single period
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = remove_headers_footers(page, margin=50)
        text += page_text

    # Define the regex patterns for start and end markers
    start_pattern = re.compile(r'\babstrak\b', re.IGNORECASE)
    end_pattern = re.compile(r'\bucapan terima kasih\b|\breferensi\b|\bdaftar pustaka\b', re.IGNORECASE)
    exclude_start_pattern = re.compile(r'\babstract\b', re.IGNORECASE)
    exclude_end_pattern = re.compile(r'\bpendahuluan\b', re.IGNORECASE)

    # Find the start and end indices for the main extraction
    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    # Set start and end indices based on matches or default to full text if not found
    start_idx = start_match.start() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)

    # Extract the relevant part of the text between "abstrak" and "referensi"/"daftar pustaka"
    if not start_match or not end_match:
        relevant_text = text
        cleaned_text = clean_text(relevant_text)
        # print(cleaned_text)
        return cleaned_text
    else:
        relevant_text = text[start_idx:end_idx]

    # Exclude the section between "abstract" and "pendahuluan"
    exclude_start_match = exclude_start_pattern.search(relevant_text)
    exclude_end_match = exclude_end_pattern.search(relevant_text)

    if exclude_start_match and exclude_end_match:
        exclude_start_idx = exclude_start_match.start()
        exclude_end_idx = exclude_end_match.end()
        relevant_text = relevant_text[:exclude_start_idx] + relevant_text[exclude_end_idx:]
    elif exclude_start_match:
        exclude_start_idx = exclude_start_match.start()
        relevant_text = relevant_text[:exclude_start_idx]
    elif exclude_end_match:
        exclude_end_idx = exclude_end_match.end()
        relevant_text = relevant_text[exclude_end_idx:]

    # Clean the extracted text
    cleaned_text = clean_text(relevant_text)

    # print(cleaned_text)
    return cleaned_text

def chunk_text(text, max_tokens=100, min_tokens=3):
    # Add space after periods if missing
    text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
    
    # Split sentences. Period, exclamation, question mark followed by space
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        sentence_length = len(tokenized_sentence)
        
        # Skip sentences shorter than min_tokens
        if sentence_length < min_tokens:
            continue
            
        # If current sentence is longer than max_tokens, split it
        if sentence_length > max_tokens:
            # If there's an existing chunk, add it first
            if current_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence into smaller chunks
            for i in range(0, len(tokenized_sentence), max_tokens):
                sentence_chunk = tokenized_sentence[i:i + max_tokens]
                chunks.append(tokenizer.convert_tokens_to_string(sentence_chunk))
            continue
            
        # If adding this sentence would exceed max_tokens
        if current_length + sentence_length > max_tokens:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            # Start new chunk with current sentence
            current_chunk = tokenized_sentence
            current_length = sentence_length
        else:
            # Add sentence to current chunk
            current_chunk.extend(tokenized_sentence)
            current_length += sentence_length
    
    # Add any remaining tokens as the last chunk
    if current_chunk:
        chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
        if chunk_text.strip():  # Only add non-empty chunks
            chunks.append(chunk_text)
            
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state # Get the embeddings from the last BERT layer
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_embedding(text):
    embeddings = []
    # Compute the embedding for a given text by chunking
    tokenized_chunks = chunk_text(text)

    for chunk in tokenized_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply mean pooling to get the chunk's pooled embedding
            pooled_output = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled_output.squeeze(0).numpy())  # Append each chunk's pooled embedding

    return embeddings  # Return list of individual chunk embeddings


def pad_embeddings(embedding1, embedding2):
    # Check if the embeddings are 2D
    if len(embedding1.shape) == 1:
        embedding1 = embedding1.reshape(1, -1)
    if len(embedding2.shape) == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    # Handle case where the embeddings have no length (empty embeddings)
    if embedding1.shape[0] == 0 or embedding2.shape[0] == 0:
        raise ValueError("One or both embeddings are empty")

    # Padding to the maximum length along the second dimension
    max_length = max(embedding1.shape[1], embedding2.shape[1])
    padding1 = np.zeros((embedding1.shape[0], max_length - embedding1.shape[1]))
    padding2 = np.zeros((embedding2.shape[0], max_length - embedding2.shape[1]))

    # Pad the embeddings
    embedding1_padded = np.hstack((embedding1, padding1))
    embedding2_padded = np.hstack((embedding2, padding2))

    return embedding1_padded, embedding2_padded