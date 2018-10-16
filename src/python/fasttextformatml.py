import csv

file_path = '../../resources/semeval2007datatrain.csv'
output_path = '../../resources/semeval2007trainfasttextml.txt'

with open(file_path) as f, open(output_path, 'w') as p:
    data = csv.DictReader(f)

    for line in data:
        newsheadlines = line['news_headlines']

        emotions = {}
        emotionlist = []
        labels = ''
        for key, value in line.items():
            if key in ['disgust', 'surprise', 'joy', 'sadness', 'anger', 'fear']:
                emotions[key] = int(value)
                if int(value) >= 50:
                    emotionlist.append(key)

        if emotionlist:
            labels = '__label__'.join(emotionlist)
            fasttextline = '__label__' + labels + ' ' + newsheadlines + '\n'
            p.write(fasttextline)





