import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


file_path_pred = '/../../resources/output_model3bigrammc.txt'
file_path_orig = '/../../semeval2007validfasttextmc.txt'

def emotion(file_path):
    with open(file_path) as f:
        emotion_list = list()
        for line in f:
            line_vec = line.split(" ")
            emotion = line_vec[0].replace("__label__", "")
            emotion_list.append(emotion)
        return emotion_list

def metrics(y_true, y_pred):

    precision, recall, f_score, _ = \
        precision_recall_fscore_support(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true, y_pred)

    print('Precision:' + str(precision) + '\nRecall:' + str(recall) +
          '\nF1 score:'+ str(f_score) + '\nAccuracy:' + str(accuracy))

def metrics_labels(y_true, y_pred):

    target_names = ['anger','disgust','fear','joy','sadness','surprise']
    print(classification_report(y_true, y_pred, target_names=target_names))

y_pred = emotion(file_path_pred)
y_true = emotion(file_path_orig)
metrics(y_true, y_pred)
metrics_labels(y_true, y_pred)
