import streamlit as st
import os
import io
import base64
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Optional
from PIL import Image
import datetime
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

class ResumeAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def get_gemini_response(self, input_prompt: str, pdf_content: List[Dict], job_description: str) -> str:
        try:
            response = self.model.generate_content([input_prompt, pdf_content[0], job_description])
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def process_pdf(self, uploaded_file) -> Optional[List[Dict]]:
        try:
            # Read PDF file
            pdf_bytes = uploaded_file.read()
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Get first page
            first_page = pdf_document[0]
            
            # Convert to image
            pix = first_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Close the PDF document
            pdf_document.close()
            
            return [{
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }]
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    def compare_resumes(self, resumes: List[Dict], job_description: str) -> Dict:
        """
        Compare multiple resumes and rank them based on job description match
        """
        comparison_prompt = """
        As an ATS (Applicant Tracking System) expert, compare the following resume against the job description.
        Provide:
        1. A match percentage (0-100)
        2. Top 3 strengths
        3. Top 3 areas for improvement
        4. Overall rank justification
        
        Format as:
        MATCH_PERCENTAGE: X
        STRENGTHS: strength1 | strength2 | strength3
        IMPROVEMENTS: improvement1 | improvement2 | improvement3
        JUSTIFICATION: your justification here
        """
        
        results = []
        for resume in resumes:
            response = self.get_gemini_response(comparison_prompt, [resume], job_description)
            if response:
                # Parse the response
                lines = response.strip().split('\n')
                result = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result[key.strip()] = value.strip()
                results.append(result)
        
        return results

class App:
    def __init__(self):
        self.analyzer = ResumeAnalyzer()
        self.prompts = {
            "resume_analysis": """
            You are an experienced Technical Human Resource Manager. Analyze the provided resume 
            against the job description and provide a detailed evaluation including:
            1. Overall alignment with the role
            2. Key strengths that match the requirements
            3. Areas for improvement
            4. Technical skills assessment
            5. Recommendations for the candidate
            
            Please format your response in a clear, structured manner.
            """,
            
            "match_percentage": """
            As an ATS (Applicant Tracking System) scanner with expertise in data science:
            1. Calculate the percentage match between the resume and job description
            2. Identify key missing keywords/skills
            3. Provide specific recommendations for improvement
            4. Share final thoughts on candidature
            
            Format the response as:
            Match Percentage: X%
            
            Missing Keywords:
            - [List keywords]
            
            Recommendations:
            - [List recommendations]
            
            Final Thoughts:
            [Your analysis]
            """
        }

    def setup_page(self):
        st.set_page_config(
            page_title="Multi-Resume ATS Expert",
            page_icon="📄",
            layout="wide"
        )
        st.markdown(
            """
            <style>
                .center-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                .full-width {
                    width: 100%;
                }
                .stProgress > div > div > div > div {
                    background-color: #4CAF50;
                }
                .comparison-table {
                    margin-top: 2rem;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            "<h1 style='text-align: center; color: white;'>📄 Multi-Resume ATS Analyzer</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center;'>Upload multiple resumes to compare them against a job description using AI.</p>",
            unsafe_allow_html=True
        )

    def run(self):
        self.setup_page()
        
        # Job Description Input
        st.markdown('<div class="center-container">', unsafe_allow_html=True)
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the job description here...",
            key="job_desc"
        )
        
        # Multiple Resume Upload
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF) (Max size: 5 MB each)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF resumes (Max size: 5 MB each)",
            key="resume_upload"
        )
        
        # Process uploaded files
        valid_files = []
        if uploaded_files:
            for file in uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                if file_size_mb > 5:
                    st.error(f"❌ File '{file.name}' exceeds the 5 MB size limit.")
                else:
                    valid_files.append(file)
                    st.success(f"✅ '{file.name}' uploaded successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Analysis buttons
        st.markdown('<div class="center-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_individual = st.button("🔍 Analyze Individual Resumes", type="primary")
        with col2:
            compare_resumes = st.button("🔄 Compare All Resumes")
        
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_individual and valid_files:
            self.analyze_individual_resumes(valid_files, job_description)
            
        if compare_resumes and len(valid_files) > 1:
            self.compare_multiple_resumes(valid_files, job_description)
        elif compare_resumes and len(valid_files) <= 1:
            st.warning("Please upload at least 2 resumes for comparison.")

    def analyze_individual_resumes(self, files, job_description):
        """Analyze each resume individually"""
        for file in files:
            st.markdown(f"### Analysis for {file.name}")
            with st.spinner(f"Analyzing {file.name}..."):
                pdf_content = self.analyzer.process_pdf(file)
                if pdf_content:
                    response = self.analyzer.get_gemini_response(
                        self.prompts["resume_analysis"],
                        pdf_content,
                        job_description
                    )
                    if response:
                        st.markdown(response)
                        
                        # Generate and allow download of individual report
                        report = self.generate_report(response, None)
                        st.download_button(
                            label=f"📄 Download Analysis Report - {file.name}",
                            data=report,
                            file_name=f"analysis_report_{file.name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"Failed to analyze {file.name}")

    def compare_multiple_resumes(self, files, job_description):
        """Compare all resumes and show ranking"""
        st.markdown("### Resume Comparison Results")
        
        with st.spinner("Comparing resumes..."):
            # Process all PDFs
            processed_resumes = []
            for file in files:
                pdf_content = self.analyzer.process_pdf(file)
                if pdf_content:
                    processed_resumes.append({
                        'name': file.name,
                        'content': pdf_content[0]
                    })
            
            # Compare resumes
            comparison_results = []
            for resume in processed_resumes:
                result = self.analyzer.get_gemini_response(
                    self.prompts["match_percentage"],
                    [resume['content']],
                    job_description
                )
                if result:
                    # Extract match percentage from result
                    try:
                        match_pct = float(result.split('%')[0].split(':')[1].strip())
                    except:
                        match_pct = 0
                        
                    comparison_results.append({
                        'name': resume['name'],
                        'match_percentage': match_pct,
                        'analysis': result
                    })
            
            # Sort by match percentage
            comparison_results.sort(key=lambda x: x['match_percentage'], reverse=True)
            
            # Display comparison table
            df = pd.DataFrame(comparison_results)
            st.markdown("#### Resume Rankings")
            st.dataframe(
                df[['name', 'match_percentage']].style.format({'match_percentage': '{:.1f}%'}),
                hide_index=True
            )
            
            # Display detailed analysis for each resume
            for idx, result in enumerate(comparison_results, 1):
                with st.expander(f"#{idx} - {result['name']} - {result['match_percentage']:.1f}%"):
                    st.markdown(result['analysis'])
            
            # Generate comparison report
            comparison_report = self.generate_comparison_report(comparison_results)
            st.download_button(
                label="📄 Download Comparison Report",
                data=comparison_report,
                file_name="resume_comparison_report.txt",
                mime="text/plain"
            )

    def generate_report(self, response: str, match_percentage: Optional[int] = None) -> str:
        """Generate a downloadable report for individual analysis"""
        report = f"ATS Resume Analysis Report\n\n"
        report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if match_percentage is not None:
            report += f"Match Percentage: {match_percentage}%\n\n"
        report += "Analysis Details:\n"
        report += response
        return report

    def generate_comparison_report(self, results: List[Dict]) -> str:
        """Generate a downloadable report for resume comparison"""
        report = "ATS Resume Comparison Report\n\n"
        report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "Rankings:\n"
        
        for idx, result in enumerate(results, 1):
            report += f"\n{idx}. {result['name']} - {result['match_percentage']:.1f}%\n"
            report += "=" * 50 + "\n"
            report += result['analysis'] + "\n"
            
        return report

if __name__ == "__main__":
    app = App()
    app.run()




    