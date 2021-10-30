import docx2txt
import gensim
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import re

nltk.download("wordnet")
nltk.download("stopwords")

def data_prep():

    # get data from csv, combine with company info
    data = pd.read_csv("F500Data.csv")

    data = data.sort_values(by="CompanyName").reset_index(drop=True)

    compID = pd.read_csv("CompanyID.csv")
    compID = compID.sort_values(by="CompanyName").reset_index(drop=True)
    compID = compID.iloc[:, :5].drop('Notes', 1)

    data = pd.merge(data.loc[:, ['CompanyName', 'xOTIOverall']], compID.loc[:, ['CompanyName', 'ID Number']],
                    on="CompanyName")

    # get company ID and mission statements from missions directory, then combine with existing data

    miss = []
    idx = []

    dir = "./Missions"

    for entry in os.scandir(dir):

        if (entry.path.endswith(".docx")):
            idx.append(re.findall(r'\d+', str(entry))[0])
            miss.append(docx2txt.process(entry))

    d = {'ID Number': idx, 'Mission Statement': miss}

    miss_df = pd.DataFrame(d)

    # Remove the Title of the documents from the text
    miss_df['Mission Statement'] = [row[row.find('\n'):] for row in miss_df['Mission Statement']]

    # Convert ID number to an int
    miss_df['ID Number'] = miss_df['ID Number'].astype(int)

    # Create a dataframe combining the mission statements with the other company information by looking at the ID Number
    data = pd.merge(data, miss_df, on="ID Number")

    # Convert xOTIOverall to float
    data['xOTIOverall'] = pd.to_numeric(data['xOTIOverall'], errors='coerce')

    # unvectorized/processed data for use with ROBERTA architectures

    # Uses gensim simple preprocess text by removing punctuation, removing unecessary characters, and vectorize
    # lower case words
    data['Mission Statement'] = [list(gensim.utils.simple_preprocess(str(row), deacc=True))
                                 for row in data['Mission Statement']]

    # apply stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['company', 'name'])

    def remove_stopwords(text):
        words = [w for w in text if w not in stop_words]
        return words

    data['Mission Statement'] = data['Mission Statement'].apply(lambda x: remove_stopwords(x))

    # Lemmatize

    lemmatizer = WordNetLemmatizer()

    def word_lemmatizer(text):
        lem_text = [lemmatizer.lemmatize(i) for i in text]
        return lem_text

    data['Mission Statement'] = data['Mission Statement'].apply(lambda x: word_lemmatizer(x))

    wc_data = data.copy()
    tm_data = data['Mission Statement'].values.tolist()
    data['Mission Statement'] = data['Mission Statement'].astype(str)

    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    # create mindset values
    Mindset = []

    count = 0
    for index, row in data.iterrows():

        if row['xOTIOverall'] <= 3.625:
            Mindset.append("Fixed")
            count = count + 1

        elif row['xOTIOverall'] <= 5.499:
            Mindset.append("Middle")

        else:
            Mindset.append("Growth")

    data['Mindset'] = Mindset

    # add mindset data to the word cloud data which is going to be used for implementing word cloud later.
    Mindset = []

    for index, row in wc_data.iterrows():

        if row['xOTIOverall'] == None:
            Mindset.append("Not Declared")

        if row['xOTIOverall'] <= 3.625:
            Mindset.append("Fixed")

        elif row['xOTIOverall'] <= 5.499:
            Mindset.append("Middle")

        else:
            Mindset.append("Growth")

    wc_data['Mindset'] = Mindset

    return data, wc_data, tm_data, stop_words
