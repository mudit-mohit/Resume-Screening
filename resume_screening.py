import PyPDF2
import docx2txt
import nltk
import tkinter as tk
from tkinter import filedialog, messagebox
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    pdf_file.close()
    return text

def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity_score = cosine_similarity(vectors)[0][1]
    return similarity_score * 100

def calculate_similarity_and_display():
    resume_file_path = resume_file_var.get()
    jd_file_path = jd_file_var.get()

    if resume_file_path.endswith('.pdf'):
        resume_text = extract_text_from_pdf(resume_file_path)
    elif resume_file_path.endswith('.docx'):
        resume_text = extract_text_from_docx(resume_file_path)
    else:
        messagebox.showerror("Error", "Unsupported file format for Resume.")
        return

    if jd_file_path.endswith('.pdf'):
        jd_text = extract_text_from_pdf(jd_file_path)
    elif jd_file_path.endswith('.docx'):
        jd_text = extract_text_from_docx(jd_file_path)
    else:
        messagebox.showerror("Error", "Unsupported file format for Job Description.")
        return
    
    preprocessed_resume_text = preprocess_text(resume_text)
    preprocessed_jd_text = preprocess_text(jd_text)

    similarity_score = calculate_similarity(' '.join(preprocessed_resume_text), ' '.join(preprocessed_jd_text))
    similarity_label.config(text="The similarity score between the resume and job description is: {:.2f}%".format(similarity_score))

def browse_resume_file():
    resume_file_path = filedialog.askopenfilename()
    resume_file_var.set(resume_file_path)

def browse_jd_file():
    jd_file_path = filedialog.askopenfilename()
    jd_file_var.set(jd_file_path)

root = tk.Tk()
root.title("Resume-Job Description Similarity Checker")

resume_file_var = tk.StringVar()
jd_file_var = tk.StringVar()

resume_label = tk.Label(root, text="Select Resume File:")
resume_label.grid(row=0, column=0, padx=5, pady=5)

resume_entry = tk.Entry(root, textvariable=resume_file_var, width=50)
resume_entry.grid(row=0, column=1, padx=5, pady=5)

browse_resume_button = tk.Button(root, text="Browse", command=browse_resume_file)
browse_resume_button.grid(row=0, column=2, padx=5, pady=5)

jd_label = tk.Label(root, text="Select Job Description File:")
jd_label.grid(row=1, column=0, padx=5, pady=5)

jd_entry = tk.Entry(root, textvariable=jd_file_var, width=50)
jd_entry.grid(row=1, column=1, padx=5, pady=5)

browse_jd_button = tk.Button(root, text="Browse", command=browse_jd_file)
browse_jd_button.grid(row=1, column=2, padx=5, pady=5)

calculate_button = tk.Button(root, text="Calculate Similarity", command=calculate_similarity_and_display)
calculate_button.grid(row=2, column=1, padx=5, pady=10)

similarity_label = tk.Label(root, text="")
similarity_label.grid(row=3, column=1, padx=5, pady=5)

root.mainloop()
