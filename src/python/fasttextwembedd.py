import io
import csv
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import contractions
import string
from nltk.corpus import stopwords
nltk.download('stopwords')

nltk.download('punkt')

emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


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

class WordEmbeddings(object):
    def __init__(self, file_path, vocab):
        self.file_path = file_path
        self.vocab = vocab
        #8including the emotions on the vocabulary, because I also need the word vectors for each emotion.
        self.vocab.extend(emotions)
        self._load()

    def _load(self):
        print('Loading vocabulary vectors from {}'.format(self.file_path))
        #9 open the word embedding file ( code from Fasttext )
        fin = io.open(self.file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        self.wb = {}
        #for each line in word vector file
        for line in fin:
            #separete in tokens (301 tokens, first is the word on the word vector, the the last 300 are the dimensions)
            tokens = line.rstrip().split(' ')
            #If the word is on the SemEval-2007 vocabulary
            if tokens[0] in self.vocab:
                #assign a vector of the 300 word vectors to the key (word)
                self.wb[tokens[0]] = list(map(float, tokens[1:]))
        print('Loaded {} vectors. Not found {} words within vectors.'.format(
            len(self.wb.keys()), len(self.vocab) - len(self.wb.keys())))
        return self.wb

    def get_vector(self, word):
        try:
            return np.array(self.wb[word])
        except:
            print('WARNING, word \'{}\' not found'.format(word))
        return np.zeros(300)

    def sentence_2_vector(self, sentence):
        vectors = [self.get_vector(token) for token in sentence]
        return np.mean([v for v in vectors], axis=0)


class Dataset():
    def __init__(self, file_path, threshold = -1):
        self.file_path = file_path
        self.threshold = threshold
        self.x = []
        self._load_dataset()
        self._load_vocab()

    def vocabulary(self):
        #7 transform the set in a list of words from the vocab
        return list(self.vocab)

    def preprocess(self, text):
        #4 pre-processing steps
        textprep = text.lower()
        textprep = contractions.fix(textprep)
        textprep = remove_pontuaction(textprep)
        textprep = remove_stopwords(textprep)
        textprep = word_tokenize(textprep)
        return textprep

    def _load_vocab(self):
        self.vocab = set()
        #5 second step: create a vocabulary of unique words from the whole corpus
        for sentence in self.x:
            self.vocab.update(sentence)
        #print how many unique words are on the volabulary
        print('Dataset vocab size: {}'.format(len(self.vocab)))

    def _load_dataset(self):
        #3 first step :open the file and for each emotion with score greather than the threshold assign the
        # headline already preprocessed to x
        with open(self.file_path) as f:
            for row in csv.DictReader(f):
                emotion_line = []
                for emotion in emotions:
                    emotion_line.append(float(row[emotion]))
                if len([e for e in emotion_line if e >= self.threshold]) > 0:
                    self.x.append(self.preprocess(row['news_headlines']))


def cos_dist(v1, v2):
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]


def normalize(result):
    summ = sum([v for k, v in result.items()])
    for k, v in result.items():
        result[k] = v/float(summ)
    return result

#1 including the path files to the semeval data set and pre trained word embeddings
dataset_file_path = '/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources/semeval2007dataunsupervised.csv'
word_embeddings_file_path = '/home/kenelly/workspaces/emotionnewsheadlines/fasttext/fastText-0.1.0/wiki-news-300d-1M.vec'
output_file = '/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources/model_unsupervised_mc_output.txt'


#2 Creating instances of dataset and wordembedding classes
dataset = Dataset(dataset_file_path)
#6 Creating instances of wordembedding classes
word_embeddings = WordEmbeddings(word_embeddings_file_path, vocab=dataset.vocabulary())

#10 for each emotion {anger, disgust, fear, joy, sadness, surprise} collect the word vector of each emotion
emotions_vector = [word_embeddings.get_vector(emotion) for emotion in emotions]

with open(output_file, 'w') as w:
    # for each sentence  in the dataset (already tokenized)
    for sentence in dataset.x:
        #get the word vector of the sentence
        sent_vec = word_embeddings.sentence_2_vector(sentence)
        #calculate the cosine similarity between the sentence word vector and each one of the six emotions, then normalize
        result = normalize(dict(zip(
            emotions,
            [cos_dist(sent_vec, em_vec) for em_vec in emotions_vector]
        )))
        #creating the file on the same format of FastText output __label__emotion score
        line = " ".join(["__label__{} {}".format(k, v) for k, v in
                sorted(result.items(), key=lambda x: x[1], reverse=True)])

        w.write(line + '\n')

