import csv

file_path = '../../resources/semeval2007datatrial.csv'
output_path = '../../resources/semeval2007trialfasttext.txt'

with open(file_path) as f, open(output_path, 'w') as p:
    data = csv.DictReader(f)

    for line in data:
        newsheadlines = line['news_headlines']

        emotions = {}
        for key, value in line.items():
            if key in ['disgust', 'surprise', 'joy', 'sadness', 'anger', 'fear']:
                emotions[key] = int(value)

        emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        emotionlabel = emotions[0][0]

        fasttextline = '__label__' + emotionlabel + ' ' + newsheadlines + '\n'

        p.write(fasttextline)


