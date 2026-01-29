# ================================
# Imports
# ================================

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer, bigrams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import regex as re


# ================================
# Global objects
# ================================

stemmer = PorterStemmer()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ================================
# Utility functions
# ================================

def stem_dict(dictionary):
    stemmed_dict = {}
    for key, value in dictionary.items():
        if len(key.split()) > 1:
            stem_key = ' '.join(stemmer.stem(i) for i in key.split())
        else:
            stem_key = stemmer.stem(key)
        stemmed_dict[stem_key] = value
    return stemmed_dict


def remove_bullets(text):
    return re.sub(
        r'^[\-\•\*]|\d+\.\s|^[a-zA-Z]$$\s|^[ivx]+\.\s',
        '',
        text,
        flags=re.MULTILINE
    )


def remove_punctuations(text):
    punctuations = """,.:;()[]{}"'•–—/"""
    pattern = str.maketrans('', '', punctuations)
    return text.translate(pattern).replace('&', 'and')


# ================================
# Resume preprocessing
# ================================

def preprocess_resume(text):
    text = remove_bullets(text)
    text = remove_punctuations(text)
    lines = [i.lower() for i in text.split('\n') if i != '']

    processed_lines = []
    for index, line in enumerate(lines):
        stemmed_line = ' '.join(stemmer.stem(i) for i in line.split())
        processed_lines.append((index, stemmed_line))

    return processed_lines


# ================================
# Heading configuration
# ================================

cue_phrases = {
    "academic background": "education",
    "academic qualifications": "education",
    "educational qualifications": "education",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "technical skills": "skills",
    "core skills": "skills",
    "skill set": "skills",
    "professional summary": "summary",
    "career objective": "summary",
    "profile summary": "summary",
    "contact details": "personal_info",
    "personal information": "personal_info",
    "academic projects": "projects",
    "key projects": "projects",
    "professional certifications": "certifications",
    "technical certifications": "certifications",
    "language proficiency": "languages",
    "languages known": "languages",
    "hobbies and interests": "interests",
    "interests and activities": "interests"
}

cue_words = {
    "education": "education",
    "experience": "experience",
    "skills": "skills",
    "summary": "summary",
    "objective": "summary",
    "projects": "projects",
    "certifications": "certifications",
    "languages": "languages",
    "interests": "interests",
    "hobbies": "interests",
    "contact": "personal_info",
    "volunteer": "others",
    "publications": "others"
}

stemmed_cue_phrases = stem_dict(cue_phrases)
stemmed_cue_words = stem_dict(cue_words)


# ================================
# Heading detection (Scan-1 & Scan-2)
# ================================

def detect_headings(resume_lines, cue_phrases, cue_words):
    heading_document = {}

    # -------- Scan 1: Bigram-based cue phrase detection --------
    all_bigrams = []
    for idx, line in resume_lines:
        tokens = [stemmer.stem(i) for i in line.split()]
        all_bigrams.extend(list(bigrams(tokens)))

    final_bigrams = [' '.join(i) for i in all_bigrams]

    for bg in final_bigrams:
        if bg in cue_phrases:
            for idx, line in resume_lines:
                if bg in line:
                    heading_document[cue_phrases[bg]] = idx
                    break

    # -------- Scan 2: Single-word cue detection --------
    for idx, line in resume_lines:
        tokens = line.split()
        if len(tokens) < 4:
            for token in tokens:
                if token in cue_words and cue_words[token] not in heading_document:
                    heading_document[cue_words[token]] = idx

    return dict(sorted(heading_document.items(), key=lambda x: x[1]))


# ================================
# Resume segmentation
# ================================

def segment_resume(resume_lines, heading_document):
    segmented_resume = {}
    headings = list(heading_document.items())

    for i in range(len(headings)):
        section, start = headings[i]
        end = headings[i + 1][1] if i < len(headings) - 1 else len(resume_lines)

        segment = [
            line for idx, line in resume_lines
            if start < idx < end
        ]
        segmented_resume[section] = segment

    return segmented_resume


# ================================
# Job Description semantic bucketing
# ================================

def bucket_job_description(jd_text):
    jd_sentences = [i.lower() for i in jd_text.split('\n') if i != '']
    jd_embeddings = model.encode(jd_sentences)

    semantic_templates = [
        "this section summarizes the role and the kind of candidate the company is seeking",
        "this section describes the tasks, duties, and responsibilities the candidate will perform in the role",
        "this section lists the mandatory technical and non-technical skills required to qualify for the role",
        "this section describes optional skills or qualifications that provide an advantage but are not mandatory"
    ]

    template_embeddings = model.encode(semantic_templates)

    buckets = {
        "role_overview": [],
        "responsibilities": [],
        "required_skills": [],
        "nice_to_have": []
    }

    for emb in jd_embeddings:
        sims = [
            cosine_similarity(emb.reshape(1, -1), t.reshape(1, -1))[0][0]
            for t in template_embeddings
        ]
        bucket_name = list(buckets.keys())[np.argmax(sims)]
        buckets[bucket_name].append(emb)

    for k in buckets:
        buckets[k] = np.mean(buckets[k], axis=0)

    return buckets


# ================================
# Resume–JD matching & scoring
# ================================

def compute_resume_score(segmented_resume, jd_buckets):
    # Create embeddings for each resume section
    resume_section_embeddings = {
        section: np.mean(model.encode(content), axis=0)
        for section, content in segmented_resume.items()
    }

    # JD → Resume section mapping
    mapping_policy = {
        'role_overview': ['summary'],
        'responsibilities': ['experience', 'projects'],
        'required_skills': ['skills', 'experience', 'projects', 'certifications'],
        'nice_to_have': ['skills', 'projects', 'certifications']
    }

    # Importance weights
    weights = {
        'required_skills': 0.45,
        'responsibilities': 0.25,
        'role_overview': 0.20,
        'nice_to_have': 0.10
    }

    maximum_sim = {}   # breakdown per JD section

    # Compute similarity section-wise
    for jd_section, jd_embedding in jd_buckets.items():
        similarities = []

        for resume_section in mapping_policy[jd_section]:
            if resume_section in resume_section_embeddings:
                sim = cosine_similarity(
                    resume_section_embeddings[resume_section].reshape(1, -1),
                    jd_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(sim)

        maximum_sim[jd_section] = max(similarities) if similarities else 0.0

    # Final weighted score
    final_weight = sum(
        maximum_sim[section] * weights[section]
        for section in maximum_sim
    )

    # ---------------- Interpretation ----------------
    print("Final resume score:", round(final_weight, 3))

    if 0.60 < final_weight <= 1:
        verdict = "Excellent match"
    elif final_weight > 0.45:
        verdict = "Strong match"
    elif final_weight > 0.33:
        verdict = "Borderline match"
    elif final_weight > 0.25:
        verdict = "Weak match"
    else:
        verdict = "Poor match"

    print("Verdict:", verdict)

    # # ---------------- Breakdown ----------------
    # print("\nBreakdown:")
    # for section, score in maximum_sim.items():
    #     print(f"{section}: {round(float(score), 3)}")

    return final_weight, verdict, maximum_sim



#================================
#Load files
#================================
def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ================================
# Main execution
# ================================

resume_text = load_text_file("resume.txt")
jd_text = load_text_file("jd.txt")

resume_lines = preprocess_resume(resume_text)

heading_document = detect_headings(
    resume_lines,
    stemmed_cue_phrases,
    stemmed_cue_words
)

segmented_resume = segment_resume(resume_lines, heading_document)

jd_buckets = bucket_job_description(jd_text)

final_score, verdict, breakdown = compute_resume_score(segmented_resume, jd_buckets)

print("Breakdown:")
for k, v in breakdown.items():
    print(k, ":",v)
