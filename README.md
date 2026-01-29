# Resume Screening Application

## Overview
This project implements a resume screening system that compares resumes against job descriptions and produces a match score. The system combines rule-based resume segmentation with semantic similarity and weighted scoring to generate an interpretable result.

## Features
- Resume segmentation using cue phrases and cue words
- Semantic bucketing of job descriptions
- Cosine similarity based matching using embeddings
- Weighted final score with clear verdict
- Section wise similarity breakdown

## How It Works
1. The resume is preprocessed and segmented into sections such as summary, experience, skills, and projects using heading detection.
2. The job description is converted into semantic buckets like role overview, responsibilities, required skills, and nice to have skills.
3. Resume sections and JD buckets are embedded and compared using cosine similarity.
4. A weighted final score is calculated and mapped to a human readable verdict.

## Usage
1. Clone the repository and install dependencies

- git clone https://github.com/Deepika081/resume-screening-system
- cd resume-screening-system
- pip install -r requirements.txt

2. Add input files

- sample_data/resume.txt  
- sample_data/jd.txt  

3. Run the script

python resume_scoring.py

The output includes the final resume score, match verdict, and a section wise similarity breakdown.

## Project Structure
experiments/  
Original exploratory and experimental code  

sample_data/  
Sample resume and job description text files  

resume_scoring.py
requirements.txt  
README.md  

## Notes
- Cue phrase and cue word dictionaries are extensible.
- Designed for text based resumes.
- PDF or DOC resumes require an additional parsing step.
- Scoring thresholds can be adjusted in the scoring logic.