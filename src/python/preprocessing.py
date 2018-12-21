import csv
import nltk
import contractions
import string
import argparse
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import LancasterStemmer, WordNetLemmatizer

#file_path = '../../resources/semeval2007datatrain.csv'
#output_path = '../../resources/semeval2007trainfasttextmlpreproc.txt'


# removing contractions
def replace_contractions(text):
    return contractions.fix(text)


# transforming to lowercase
def lowercase(text):
    return text.lower()


# removing pontuaction
def remove_pontuaction(text):
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text


# removing stop words
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    not_stopwords = ['against', 'above', 'below', 'up',
                     'down', 'no', 'nor', 'not']
    final_stopwords = [word for word in stop_words if word not in not_stopwords]
    headlines_words = text.split()
    filtered_headlines = [word for word in headlines_words if word not in final_stopwords]
    return " ".join(filtered_headlines)


def preprocessing(file_path, output_path, type):
    with open(file_path) as f, open(output_path, 'w') as p:
        data = csv.DictReader(f)

        for line in data:
            newsheadlines = line['news_headlines']
            newsheadlines = lowercase(newsheadlines)
            newsheadlines = replace_contractions(newsheadlines)
            newsheadlines = remove_pontuaction(newsheadlines)
            newsheadlines = remove_stopwords(newsheadlines)

            if type == 'ml':
                emotions = {}
                emotionlist = []

                for key, value in line.items():
                    if key in ['disgust', 'surprise', 'joy', 'sadness', 'anger', 'fear']:
                        emotions[key] = int(value)
                        if int(value) >= 50:
                            emotionlist.append(key)

                if emotionlist:
                    labels = ' '.join(['__label__' + e for e in emotionlist])
                    fasttextline = labels + ' ' + newsheadlines + '\n'
                    p.write(fasttextline)

            elif type == 'mc':
                emotions = {}
                for key, value in line.items():
                    if key in ['disgust', 'surprise', 'joy', 'sadness', 'anger', 'fear']:
                        emotions[key] = int(value)

                emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                emotionlabel = emotions[0][0]

                fasttextline = '__label__' + emotionlabel + ' ' + newsheadlines + '\n'
                p.write(fasttextline)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("SemEval_2007 Preprocessing")
    parser.add_argument('--file_path', help='Input file path')
    parser.add_argument('--output_path', help='Output file path')
    parser.add_argument('--type', help='Multilabel - ml or Multiclass - mc')
    args = parser.parse_args()

    preprocessing(args.file_path, args.output_path, args.type)



