from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer
import regex as re

# Preprocessing

def remove_bullets(text):
    cleaned_text = re.sub(r'^[\-\•\*]|\d+\.\s|^[a-zA-Z]$$\s|^[ivx]+\.\s', '', text, flags=re.MULTILINE)
    return cleaned_text

text = """
Mystery Penguin
example@email.com
(555)123-4567
Edgar Springs, MO
Summary
Competent software engineer, able to work effectively in a fast-paced, agile environment, and passionate about developing software architecture for primarily web applications.
Experience
Software Engineer
BlueTech Software • Los Angeles, California
July 2019 - Present
Researched and created a new application to be used by the company, resulting in a 1.3x increase in sales
Created a new software development framework to develop applications faster and more efficiently
Developed the process of testing and recording the results of the test, thereby increasing the number of tests completed per day by 10%
Developed a new technique for reducing the time it takes to fix crashes by 20%
Web Developer
Mezzanine • Los Angeles, California
May 2018
Developed a collaborative environment for all team members, creating a strong sense of community and empowerment
Designed a new web application to increase collaboration among design, engineering and sales teams, resulting in a 10% increase in revenue in one quarter
Built a custom web application for a major client, successfully delivering an integrated CRM system in just 2 weeks
Maintained a clean, professional and creative web presence for the company to convey the professional image of the organization
Used Adobe InDesign to design, edit and layout the company's print & digital collateral
Software Engineer
Geeks 4 Nerds • Los Angeles, California
October 2016
Designed and implemented a new cloud-based mobile POS platform, resulting in a 50% increase in customer retention
Collaborated with hardware and software development teams to ensure seamless integration of the new POS system
Continued to develop specialty skills related to his expertise in mobile POS and payment technologies
Developed and continuously improved an internal knowledge base of best practices for software development
Skills
C, Java, Sdlc, Software Development, Linux, C#, Communication Skills
Education
Bachelor's Degree in Computer Science
Zirkel College • Los Angeles, California
"""
new_text = remove_bullets(text)
l = [i.lower() for i in new_text.split('\n') if i!= '']
new_list = []
for index,line in enumerate(l):
    new_list.append((index,line))
#print(new_list)

cue_phrases = {
'personal info':['contact details', 'personal information'],
'summary':['professional summary', 'career objective','profile summary'],
'education':['academic background', 'academic qualifications', 'educational qualifications'],
'experience': ['work experience', 'professional experience', 'employment history'],
'skills':['technical skills', 'core skills','skill set'],
'projects': ['academic projects', 'key projects'],
'certifications': ['professional certifications', 'technical certifications'],
'languages': ['language proficiency', 'languages known'],
'interests': ['hobbies and interests', 'interests and activities']
}
# list_cue_phrase = [i for i in initial_cue_phrases.split('\n')]
# cue_phrase = [i.split(',') for i in list_cue_phrase]

# # print(cue_phrase)
# stemming = PorterStemmer()
# stemmed_list = stemming.stem(initial_cue_phrases)

# print(stemmed_list)

cue_words = {
    'education': ['education'],
    'experience': ['experience'],
    'skills': ['skills'],
    'summary': ['summary', 'objective'],
    'projects': ['projects'],
    'certifications': ['certifications'],
    'languages': ['languages'],
    'interests': ['interests', 'hobbies'],
    'personal info': ['contact'],
    'others': ['volunteer', 'publications']
}