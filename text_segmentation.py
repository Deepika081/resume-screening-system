from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer,bigrams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import regex as re

def stem_dict(dictionary):
    stemmed_dict = {}
    stemmer = PorterStemmer()
    for key,value in dictionary.items():
        if len(key.split())>1:
            l = key.split()
            b = [stemmer.stem(i) for i in l]
            c = ' '.join(b)
            stem_key = c
        else:
            stem_key = stemmer.stem(key)
        stemmed_dict[stem_key] = value
    return stemmed_dict

# Preprocessing

def remove_bullets(text):
    cleaned_text = re.sub(r'^[\-\•\*]|\d+\.\s|^[a-zA-Z]$$\s|^[ivx]+\.\s', '', text, flags=re.MULTILINE)
    return cleaned_text

def remove_punctuations(text):
    punctuations = """,.:;()[]{}"'•–—/"""
    pattern = str.maketrans('', '',punctuations)
    cleaned_text = (text.translate(pattern)).replace('&','and')
    return cleaned_text

text = """
Summary:
Frontend-focused engineer with experience in building responsive user interfaces.

Experience:
Frontend Developer
CreativeApps Studio
June 2021 – Present
- Developed user interfaces using HTML, CSS, and JavaScript
- Built reusable UI components using React
- Collaborated with designers to improve user experience
- Integrated frontend components with backend APIs

Projects:
- Created a portfolio website using React and Tailwind CSS
- Built a dashboard for analytics visualization

Skills:
JavaScript, React, HTML, CSS, UI/UX, Figma, Git

Education:
Bachelor’s degree in Information Technology

"""

stemmer = PorterStemmer()


text_without_bullets = remove_bullets(text)
text_without_punctuations = remove_punctuations(text_without_bullets)
l = [i.lower() for i in text_without_punctuations.split('\n') if i!= '']
#print(l)
new_list = []
for index,line in enumerate(l):
    new_list.append((index,' '.join(stemmer.stem(i) for i in line.split())))
#print(new_list)

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

stemmed_cue_phrase = stem_dict(cue_phrases)
stemmed_cue_words = stem_dict(cue_words)


# Scan-1

bi_gram = []

for i in new_list:
    line = remove_punctuations(i[1])
    #tokenize and stemming
    stemmed_line = [stemmer.stem(i) for i in line.split()]
    # print(stemmed_line)
    # creating bigram
    bi_gram_line = list(bigrams(stemmed_line))
    bi_gram.extend(bi_gram_line)

final_bigram = [' '.join(i) for i in bi_gram]
heading_document = {}
temp_list = []
for i in final_bigram:
    if i in stemmed_cue_phrase.keys():
        final_list = []
        value = stemmed_cue_phrase[i]
        final_list.append(value)
        tup = [j for j in new_list if i in j]
        final_list.append(tup)
        temp_list.append(final_list)

for i in temp_list:
    heading_document[i[0]] = i[1][0][0]

#print(heading_document)

# Scan-2

for i in new_list:
    line = i[1].split()
    if len(line)<4:
        for j in line:
            if j in stemmed_cue_words.keys() and stemmed_cue_words[j] not in heading_document.keys():
                heading_document[stemmed_cue_words[j]] = i[0]

ordered_heading_document = {
    k:v for k,v in sorted(heading_document.items(), key=lambda item:item[1])
}

#print(ordered_heading_document)

segmented_resume = {}
indexes = [v for k,v in ordered_heading_document.items()]
#print(indexes)
list_of_hd = list(ordered_heading_document.items())

for i in range(len(list_of_hd)):
    start = list_of_hd[i][1]
    end = list_of_hd[i+1][1] if (i<len(list_of_hd)-1) else len(new_list)
    #print(start,end)
    segment = [j[1] for j in new_list if (start<j[0] and j[0]<end)]
    segmented_resume[list_of_hd[i][0]] = segment
# for key,index in zip(list(ordered_heading_document.items())):
#     end = index[i+1] if (i<len(indexes)-1) else len(new_list)
#     # if (i<len(indexes)-1):   
#     #    end = indexes[i+1]
#     # else:
#     #    end = len(new_list)
#     start = indexes[i]
#     print(start,end)
#     segment = [j[1] for j in new_list if (start<j[0] and j[0]<end)]
#     segmented_resume[key] = segment
#print(segmented_resume)

# Job Description

jd = """
We are looking for a Software Development Engineer to join our backend team.

Role Overview:
You will be responsible for building scalable backend services and APIs that support millions of users.

Responsibilities:
- Design, develop, and maintain backend services
- Write clean, efficient, and testable code
- Collaborate with frontend and product teams
- Optimize applications for performance and scalability

Required Skills:
- Strong proficiency in Python or Java
- Experience with REST APIs and backend frameworks
- Solid understanding of data structures and algorithms
- Experience with SQL or NoSQL databases
- Familiarity with Linux and version control systems

Nice to Have:
- Experience with cloud platforms (AWS/GCP)
- Knowledge of Docker or Kubernetes
- Prior experience in system design

"""

JD_sentences = [i.lower() for i in jd.split('\n') if i!='']
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(JD_sentences)
JD_embeddings = {}
for i in range(len(embeddings)):
    JD_embeddings[i] = embeddings[i]
semantic_bucket = [
    "this section summarizes the role and the kind of candidate the company is seeking",
    "this section describes the tasks, duties, and responsibilities the candidate will perform in the role",
    "this section lists the mandatory technical and non-technical skills required to qualify for the role",
    "this section describes optional skills or qualifications that provide an advantage but are not mandatory",
    ]
semantic_emedding = model.encode(semantic_bucket)
JD_bucket = {
    "role_overview": [],
    "responsibilities": [],
    "required_skills": [],
    "nice_to_have": []
}

for k,v in JD_embeddings.items():
    sim1 = cosine_similarity(v.reshape(1,-1),semantic_emedding[0].reshape(1,-1))
    sim2 = cosine_similarity(v.reshape(1,-1),semantic_emedding[1].reshape(1,-1))
    sim3 = cosine_similarity(v.reshape(1,-1),semantic_emedding[2].reshape(1,-1))
    sim4 = cosine_similarity(v.reshape(1,-1),semantic_emedding[3].reshape(1,-1))
    max_sim = max([sim1,sim2,sim3,sim4])
    if max_sim == sim1:
        JD_bucket['role_overview'].append(k)
    elif max_sim == sim2:
        JD_bucket['responsibilities'].append(k)
    elif max_sim == sim3:
        JD_bucket['required_skills'].append(k)
    else:
        JD_bucket['nice_to_have'].append(k)

print(JD_bucket)

for k,v in JD_bucket.items():
    temp_embeding = []
    for i in v:
        temp_embeding.append(JD_embeddings[i])
    JD_bucket[k] = np.mean(temp_embeding,axis=0)

resume_section_embeddings = {}

for k,v in segmented_resume.items():
    segment_embedding = model.encode(v)
    resume_section_embeddings[k] = np.mean(segment_embedding,axis=0)

print(resume_section_embeddings['summary'].shape)

# mapping JD_bucket -> resume_sections

mapping_policy = {
'role_overview' : ['summary'],
'responsibilities' : ['experience', 'projects'],
'required_skills' : ['skills', 'experience', 'projects','certifications'],
'nice_to_have' : ['skills', 'projects', 'certifications']
}

maximum_sim = {}

for k,v in JD_bucket.items():
    maximum_sim[k] = max([cosine_similarity(resume_section_embeddings[i].reshape(1,-1),JD_bucket[k].reshape(1,-1))for i in mapping_policy[k] if i in resume_section_embeddings.keys()])[0][0]

print('maximum similarity: ', maximum_sim)
weights = {
    'required_skills': 0.45,
    'responsibilities': 0.25,
    'role_overview': 0.20,
    'nice_to_have': 0.10
}

final_weight = 0

for k,v in maximum_sim.items():
    #print(type(v))
    temp_sum = v*weights[k]
    final_weight += temp_sum

print("Final resume score: ", final_weight)

if final_weight>0.60 and final_weight<1:
    print('Excellent match')
elif final_weight>0.45:
    print('Strong match')
elif final_weight>0.33:
    print('Borderline match')
elif final_weight>0.25:
    print('Weak match')
else:
    print('Poor match')

print('Breakdown')
for k,v in maximum_sim.items():
    print(k,':',float(v))