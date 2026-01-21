from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer,bigrams
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
Mystery Penguin
example@email.com
(555)123-4567
Edgar Springs, MO
Summary
Competent software engineer, able to work effectively in a fast-paced, agile environment, and passionate about developing software architecture for primarily web applications.
Work Experience
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
Academic Background
Bachelor's Degree in Computer Science
Zirkel College • Los Angeles, California
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

print(ordered_heading_document)

segmented_resume = {}
indexes = [v for k,v in ordered_heading_document.items()]
print(indexes)
list_of_hd = list(ordered_heading_document.items())

for i in range(len(list_of_hd)):
    start = list_of_hd[i][1]
    end = list_of_hd[i+1][1] if (i<len(list_of_hd)-1) else len(new_list)
    print(start,end)
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
print(segmented_resume)