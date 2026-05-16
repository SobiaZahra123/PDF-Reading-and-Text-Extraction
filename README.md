# NLP Assignment: PDF Text Processing and Feature Extraction
 # Overview
This project implements complete Natural Language Processing (NLP) text processing on a real PDF document with 100+ pages
from Google Scholar. The assignment demonstrates practical application of:

 1.PDF text extraction from academic papers.
2. Comprehensive text preprocessing using Regular Expressions (Regex).
3. Tokenization, stop word removal, stemming, and lemmatization.
4. Feature extraction techniques: One-Hot Encoding and TF-IDF.
5. Interactive data visualization using Plotly.
6. Complete NLP pipeline from raw text to analysis.

#  Learning Objectives
1. How to extract and process text from large PDF documents
2. Apply regex patterns for text cleaning
3. Understand the difference between stemming and lemmatization
4. Implement feature extraction for NLP tasks
5. Create interactive visualizations for data analysis

# Features     

Q1(a): PDF Reading and Text Extraction
Read PDF files using pypdf library
Extract text from all 120 pages
Display total pages and sample text
Character count statistics
Q1(b): Text Preprocessing
Lowercase Conversion: Convert all text to lowercase
Remove Numbers: Using regex pattern r'\d+'
Remove Special Symbols: Using regex pattern r'[^a-zA-Z\s]'
Remove Extra Spaces: Using regex pattern r'\s+'
Tokenization: Split text into individual words
Stop Word Removal: Remove common English words
Stemming: Apply Porter Stemmer algorithm
Lemmatization: Apply WordNet Lemmatizer
Statistics: Display counts of stop words and valid words
Q1(c): Feature Extraction
One-Hot Encoding
Convert words into binary matrix representation
Tabular display of encoding results
Useful for machine learning algorithms
TF-IDF (Term Frequency-Inverse Document Frequency)
Calculate importance of words in documents
Feature name extraction
Tabular display of TF-IDF values
Identify most relevant words
Q1(d): Visualization
TF-IDF Scatter Plot using Matplotlib
Word index on X-axis
TF-IDF scores on Y-axis
Word labels displayed on plot
Proper title and axis labels
Saved as PNG image
#  PDF Reading and Text Extraction
1. Selected PDF with 100+ pages from Google Scholar
2. Read PDF using PyPDF2/PyMuPDF
3. Extracted text from all pages
4. Displayed total number of pages
5. Showed sample extracted text

#  Text Preprocessing with Regex
1. Converted text to lowercase
2. Removed numbers using Regex: \d+
3. Removed special symbols using Regex: [^a-zA-Z\s]
4. Removed extra spaces using Regex: \s+
5. Removed punctuation using Regex: [^\w\s]
6. Tokenized text into words
7. Applied stop word removal
8. Displayed stop word count
9. Displayed valid word count after removal
10. Applied stemming (Porter Stemmer)
11. Applied lemmatization (WordNet Lemmatizer)

#  Feature Extraction
1. One-Hot Encoding on cleaned valid words (tabular form)
2. TF-IDF on cleaned text
3. Displayed TF-IDF feature names
4. Displayed TF-IDF values in tabular form

#  TF-IDF Scatter Plot Using Plotly
1. Created scatter plot using Plotly only
2. Displayed words and their TF-IDF scores
3. Included proper title, x-axis label, y-axis label
4. Interactive visualization with hover effects

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

