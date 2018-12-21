from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


#file path predicted labels
file_path_pred = '/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources/model_unsupervised_ml_output.txt'

#file path true labels
file_path_true = '/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources/semeval2007testunsupervisedml.txt'

#function to collect the predicted and true labels from the files
def emotion_pred(file_path_pred):
    with open(file_path_pred) as f:
        emotion_list_pred = list()
        for line in f:
            line_vec = line.split(" ")
            emotion = line_vec[0].replace("__label__", "")
            emotion_list_pred.append(emotion)
        return emotion_list_pred

#function to collect the predicted and true labels from the files
def emotions(file_path_pred, file_path_true):
    pred = emotion_pred(file_path_pred)#src/python/evaluationml.py:24
    pred_final = []
    true_final = []
    i=0
    with open(file_path_true) as p:
        for line in p:
            line_vec = line.split(" ")
            for label in line_vec:
                if label.startswith('__label__'):
                    emotion_true = label.replace("__label__", "")
                    true_final.append(emotion_true)
                    pred_final.append(pred[i])
            i+=1

    return pred_final, true_final


y_pred, y_true = emotions(file_path_pred, file_path_true)

#function to calculate multilabel metrics
def metrics(y_true, y_pred):

    precision, recall, f_score, _ = \
        precision_recall_fscore_support(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true, y_pred)

    print('Precision:' + str(precision) + '\nRecall:' + str(recall) +
          '\nF1 score:'+ str(f_score) + '\nAccuracy:' + str(accuracy))

#function to calculate the metrics by label
def metrics_labels(y_true, y_pred):

    target_names = ['anger','disgust','fear','joy','sadness','surprise']
    print(classification_report(y_true, y_pred, target_names=target_names))



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '17f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

target_names = ['anger','disgust','fear','joy','sadness','surprise']
metrics(y_true, y_pred)
metrics_labels(y_true, y_pred)
confusionmatrix = confusion_matrix(y_true, y_pred, labels=['anger','disgust','fear','joy','sadness','surprise'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(confusionmatrix, classes=target_names,
                      title='Confusion matrix multi-label')
plt.show()
print(confusionmatrix)
