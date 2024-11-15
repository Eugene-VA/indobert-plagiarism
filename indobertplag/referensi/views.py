from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from plagiarism.models import Reference
from plagiarism.utils import extract_text_from_pdf, compute_embedding
import os
import numpy as np

# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('admin_dashboard')  # Redirect to admin_dashboard instead of manage
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'login.html')

# Logout view
def logout_view(request):
    logout(request)
    return redirect('login')

# Admin dashboard view
@login_required
def admin_dashboard(request):
    references = Reference.objects.all().order_by('-title')
    return render(request, 'manage.html', {'references': references})

@login_required
def add_reference(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        authors = request.POST.get('authors')
        pdf = request.FILES.get('pdf')

        if pdf:
            # Check if the title exists in the database (case-insensitive)
            existing_reference = Reference.objects.filter(title__iexact=title).first()
            if existing_reference:
                messages.error(request, 'A reference with this title already exists. Please choose a different title.')
                return redirect('add_reference')

            fs = FileSystemStorage()
            file_path = fs.path(pdf.name)

            # If the file exists in the media folder but not in the database, delete the old file
            if fs.exists(pdf.name):
                os.remove(file_path)  # Remove the old file from the media folder

            # Save the new file and extract its content
            filename = fs.save(pdf.name, pdf)
            pdf_path = fs.path(filename)
            content = extract_text_from_pdf(pdf_path)
            embeddings = compute_embedding(content)
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            # Create the new reference in the database
            Reference.objects.create(title=title, authors=authors, content=content, embeddings=embeddings_list)

            messages.success(request, 'Reference added successfully!')
            return redirect('admin_dashboard')

    return render(request, 'add_reference.html')

# Delete reference view
@login_required
def delete_reference(request, reference_id):
    reference = Reference.objects.get(id=reference_id)
    reference.delete()
    return redirect('admin_dashboard')  # Correctly redirect after deleting

@login_required
def recalculate_embeddings(request):
    references = Reference.objects.all()

    for ref in references:
        embeddings = compute_embedding(ref.content) # Get the list of embeddings for each sentence
        embeddings_as_lists = [embedding.tolist() for embedding in embeddings] # Convert each numpy array in embeddings to a Python list for JSON storage
        ref.embeddings = embeddings_as_lists # Update the reference's embeddings field
        ref.save()  # Save changes to the database entry

    return redirect('admin_dashboard')