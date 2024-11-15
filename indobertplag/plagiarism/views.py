from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib import messages
from sklearn.metrics.pairwise import cosine_similarity
from .models import Reference
from .utils import extract_text_from_pdf, compute_embedding, chunk_text, pad_embeddings
import numpy as np
import os
similarity_threshold = 0.7

def index(request):
    return render(request, 'index.html')

def upload_pdf(request):
    if request.method == 'POST' and request.FILES.get('pdf'):
        pdf = request.FILES['pdf']
        fs = FileSystemStorage()

        # Delete the old file if it already exists
        if fs.exists(pdf.name):
            old_file_path = fs.path(pdf.name)
            os.remove(old_file_path)

        # Save the new file
        filename = fs.save(pdf.name, pdf)
        file_path = fs.path(filename)

        # Extract and chunk the uploaded PDF text into sentences
        text = extract_text_from_pdf(file_path)
        input_embeddings = np.array(compute_embedding(text)) 

        # Retrieve all reference sentences and their embeddings from the database
        reference_sentences = Reference.objects.all()

        plagiarized_count = 0  # Count sentences flagged as plagiarized
        combined_results = {}
        input_sentences = chunk_text(text)
        total_sentences = len(input_sentences)

        # Prepare reference embeddings outside the loop for vectorized operations
        reference_embeddings = [
            np.array([np.array(e) for e in ref.embeddings]) for ref in reference_sentences
        ]
        
        print("Shape of input_embeddings:",input_embeddings.shape)
        print("Shape of reference_embeddings:",len(reference_embeddings))

        # Process each input sentence and compare with reference sentences
        for i, input_embedding in enumerate(input_embeddings):
            max_similarity, max_reference, input_sentence = process_embedding(
                input_embedding, reference_embeddings, reference_sentences, input_sentences[i]
            )

            # Update results if similarity exceeds threshold
            if max_similarity > similarity_threshold:
                plagiarized_count += 1
                if max_reference.title in combined_results:
                    combined_results[max_reference.title]['total_similarity'] += max_similarity
                    combined_results[max_reference.title]['count'] += 1
                    combined_results[max_reference.title]['input_sentences'].append(input_sentence)
                else:
                    combined_results[max_reference.title] = {
                        'count': 1,
                        'total_similarity': max_similarity,
                        'input_sentences': [input_sentence],
                        'author': max_reference.authors
                    }

        # Prepare the final results list
        results = [
            {
                'reference': title,
                'similarity': f"{(data['count'] / total_sentences) * 100:.2f}%",
                'similarity_score': data['count'] / total_sentences,
                'author': data['author']
            }
            for title, data in combined_results.items()
        ]

        # Calculate weighted score
        weighted_score = (plagiarized_count / total_sentences) * 100 if total_sentences > 0 else 0
        score_label = (
            "Non-plagiat" if weighted_score < 10 else
            "Plagiat ringan" if 10 <= weighted_score < 30 else
            "Plagiat berat"
        )

        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        context = {
            'uploaded_file_url': fs.url(filename),
            'weighted_score': f"{weighted_score:.2f}% ({score_label})",
            'similarities': results,
            'filename': filename
        }

        return render(request, 'upload_success.html', context)
    else:
        messages.error(request, "No PDF file uploaded.")
        return redirect('index')

def process_embedding(input_embedding, reference_embeddings, reference_sentences, input_sentence):
    max_similarity = 0
    max_reference = None

    for ref_embeddings_array, ref in zip(reference_embeddings, reference_sentences):
        # Pad the input and reference embeddings once
        input_embedding_padded, ref_embeddings_padded = pad_embeddings(input_embedding, ref_embeddings_array)

        # Print the shapes after padding
        # print("Shape of input_embedding_padded:", input_embedding_padded.shape)
        # print("Shape of ref_embeddings_padded:", ref_embeddings_padded.shape)

        # Compute cosine similarity in a vectorized way
        similarity_scores = cosine_similarity(input_embedding_padded.reshape(1, -1), ref_embeddings_padded).flatten()
        # print("Shape of reshaped input embedding:", input_embedding_padded.reshape(1, -1).shape)
        # # Print the shape of similarity_scores
        # print("Shape of similarity_scores:", similarity_scores.shape)

        # Find max similarity and track highest similarity reference
        max_sim_score = np.max(similarity_scores)
        if max_sim_score > max_similarity:
            max_similarity = max_sim_score
            max_reference = ref  # Track the actual Reference object

    return max_similarity, max_reference, input_sentence

def delete_file_and_redirect(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    # Check if the file exists before attempting to delete it
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Redirect to the index page after deletion
    return redirect('index')
