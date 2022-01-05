import nltk
import re
import unidecode
import sys
import string

def preprocessing_text(text):

    text = text.lower()
    text = unidecode.unidecode(text)
    
    text = re.sub('\[.*?\]', '', text) 
    
    text_without_url = re.sub('https?://\S+|www\.\S+', '', text) 
    
    text_without_tag = re.sub('<.*?>+', '', text_without_url)
    
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text_without_tag) 
    
    text = re.sub('\n', '', text) 
    text = re.sub('\w*\d\w*', '', text) 
    
    return text

def preprocessing(data):

    data.text = data.text.apply(preprocessing_text)

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    data.text = data.text.apply(tokenizer.tokenize)

    nltk.download('stopwords')

    data.text = data.text.apply(lambda text: [w for w in text if w not in nltk.corpus.stopwords.words('english')])

    data.text = data.text.apply(lambda text: ' '.join(text))

    return data
