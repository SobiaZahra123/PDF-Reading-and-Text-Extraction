 # Overview
This project implements complete Natural Language Processing (NLP) text processing on a real PDF document with 100+ pages
from Google Scholar. The assignment demonstrates practical application of:

✅ PDF text extraction from academic papers
✅ Comprehensive text preprocessing using Regular Expressions (Regex)
✅ Tokenization, stop word removal, stemming, and lemmatization
✅ Feature extraction techniques: One-Hot Encoding and TF-IDF
✅ Interactive data visualization using Plotly
✅ Complete NLP pipeline from raw text to analysis

#  Learning Objectives
1. How to extract and process text from large PDF documents
2. Apply regex patterns for text cleaning
3. Understand the difference between stemming and lemmatization
4. Implement feature extraction for NLP tasks
5. Create interactive visualizations for data analysis

# Repository Structure
text
├── 📄 README.md                          
├── 📓 nlp_feature_Engineering.py          
├── 📄 academic_paper.pdf                
├── 📊 tfidf_scatter_plot.html           
├── 📊 tfidf_bubble_chart.html           
├── 📈 tfidf_results.csv                 
├── 📈 onehot_encoding_results.csv        
├── 📝 preprocessing_results.txt         
├── 📦 requirements.txt                   
├── 🐍 setup.sh                           
├── 🐍 setup.bat                         
├── 📸 screenshots/                       #
│   ├── 01_pdf_reading.png
│   ├── 02_total_pages.png
│   ├── 03_sample_text.png
│   ├── 04_regex_preprocessing.png
│   ├── 05_stopword_count.png
│   ├── 06_valid_word_count.png
│   ├── 07_stemming_output.png
│   ├── 08_lemmatization_output.png
│   ├── 09_one_hot_encoding.png
│   ├── 10_tfidf_output.png
│   ├── 11_plotly_scatter.png
│   ├── 12_github_repo.png
│   └── 13_kaggle_notebook.png
└── 🔧 .gitignore                       
#  PDF Reading and Text Extraction
✅ Selected PDF with 100+ pages from Google Scholar
✅ Read PDF using PyPDF2/PyMuPDF
✅ Extracted text from all pages
✅ Displayed total number of pages
✅ Showed sample extracted text

#  Text Preprocessing with Regex
✅ Converted text to lowercase
✅ Removed numbers using Regex: \d+
✅ Removed special symbols using Regex: [^a-zA-Z\s]
✅ Removed extra spaces using Regex: \s+
✅ Removed punctuation using Regex: [^\w\s]
✅ Tokenized text into words
✅ Applied stop word removal
✅ Displayed stop word count
✅ Displayed valid word count after removal
✅ Applied stemming (Porter Stemmer)
✅ Applied lemmatization (WordNet Lemmatizer)

#  Feature Extraction
✅ One-Hot Encoding on cleaned valid words (tabular form)
✅ TF-IDF on cleaned text
✅ Displayed TF-IDF feature names
✅ Displayed TF-IDF values in tabular form

#  TF-IDF Scatter Plot Using Plotly
✅ Created scatter plot using Plotly only
✅ Displayed words and their TF-IDF scores
✅ Included proper title, x-axis label, y-axis label
✅ Interactive visualization with hover effects

#  How to Run the Code
Option 1: Run Locally 
Prerequisites
bash
# Python 3.14
python --version

# Step-by-Step Code Execution Guide
Step 1: Import Libraries
python
import PyPDF2
import re
import nltk
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
Step 2: Download 100+ Page PDF from Google Scholar

# PDF URL from Google Scholar (100+ pages)
pdf_url =  https://arxiv.org/pdf/2303.08774.pdf
<img width="1406" height="736" alt="Screenshot 2026-05-16 104445" src="https://github.com/user-attachments/assets/e1c331d2-df30-4711-8fa0-908854c2a134" />
<img width="1388" height="740" alt="Screenshot 2026-05-16 104412" src="https://github.com/user-attachments/assets/ab46103d-3c18-4db9-99dc-a2b4d4afd6e2" />

