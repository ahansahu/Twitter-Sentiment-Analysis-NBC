import math
import numpy as np
import re
import csv
import nltk
from nltk.stem import PorterStemmer
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer


class classifier:
    Dict = {}
    Num_Pos = 0
    Num_Neg = 0
    Prob_Pos = 0
    Prob_Neg = 0

    def __init__(self):
        pass

    def read_training_data(self):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        with open('train.csv', newline='', encoding="utf8") as csvfile:
            data = list(csv.reader(csvfile))

        ratings = []
        tweet = []

        for i in range(1, len(data)):
            data[i][2] = data[i][2].encode('ascii', 'ignore').decode('utf-8')
            line = data[i][2]
            line = line.replace('#', 'hash_')
            line = line.replace('@user', '')
            line = re.sub(r'[^\w\s]', '', line).lower()
            line = re.sub(r'\d+', '', line)
            spl = line.split(' ')
            spl = [stemmer.stem(j) for j in spl if len(j) > 3]
            #spl = [lemmatizer.lemmatize(j) for j in spl if len(j) > 3]

            ratings.append(data[i][1])
            tweet.append(spl)

        return [ratings, tweet]

    def read_test_data(self):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        with open('test.csv', newline='', encoding="utf8") as csvfile:
            data = list(csv.reader(csvfile))

        id = []
        tweet = []

        for i in range(1, len(data)):
            data[i][1] = data[i][1].encode('ascii', 'ignore').decode('utf-8')
            line = data[i][1]
            line = line.replace('#', 'hash_')
            line = line.replace('@user', '')
            line = re.sub(r'[^\w\s]', '', line).lower()
            line = re.sub(r'\d+', '', line)
            spl = line.split(' ')
            spl = [stemmer.stem(j) for j in spl if len(j) > 3]
            #spl = [lemmatizer.lemmatize(j) for j in spl if len(j) > 3]

            id.append(data[i][0])
            tweet.append(spl)

        return [id, tweet]

    def train_NBC(self):
        stop = ["a", "but", "about", "by", "above", "cant", "after", "cannot", "again", "could", "against", "couldnt",
                "all", "am", "did", "an", "didnt", "and", "do", "any", "does", "are", "doesnt", "arent", "doing", "as",
                "dont", "at", "down", "be", "during", "because", "each", "been", "few", "before", "for", "being",
                "from", "below", "further", "between", "had", "both", "hadnt", "has"]
        # most_freq = ["day", "amp", "u", "im", "time"]

        output = self.read_training_data()
        sentiment = output[0]
        tweets = output[1]

        # onlyneg = []
        # onlypos = []
        # for i in range(len(sentiment)):
        #     if sentiment[i] == "1":
        #         onlyneg.append(tweets[i])
        #     else:
        #         onlypos.append(tweets[i])
        #
        # more_neg = resample(onlyneg, replace=True, n_samples=29720, random_state=123)
        #
        # tweets = onlypos + more_neg
        # sentiment = ["0"]*29720 + ["1"]*29720


        for i in range(len(sentiment)):
            self.Prob_Neg += 1
            self.Prob_Pos += 1

            for j in tweets[i]:
                if j not in (stop):
                    if j in self.Dict:
                        if sentiment[i] == '1':
                            word = self.Dict.get(j)
                            number = word.get('N')
                            self.Dict[j]['N'] = number + 1
                            self.Num_Neg += 1
                        else:
                            word = self.Dict.get(j)
                            number = word.get('P')
                            self.Dict[j]['P'] = number + 1
                            self.Num_Pos += 1
                    else:
                        self.Dict[j] = {'P': 0, 'N': 0}

        sum = self.Prob_Pos + self.Prob_Neg
        self.Prob_Neg = self.Prob_Neg/sum
        self.Prob_Pos = self.Prob_Pos/sum


    def Predict(self):
        stop = ["a", "but", "about", "by", "above", "cant", "after", "cannot", "again", "could", "against", "couldnt",
                "all", "am", "did", "an", "didnt", "and", "do", "any", "does", "are", "doesnt", "arent", "doing", "as",
                "dont", "at", "down", "be", "during", "because", "each", "been", "few", "before", "for", "being",
                "from", "below", "further", "between", "had", "both", "hadnt", "has"]
        # most_freq = ["day", "amp", "u", "im", "time"]

        output = self.read_test_data()
        ids = output[0]
        tweets = output[1]
        sentiment = []
        decided = False

        for i in range(len(ids)):
            Pos_Prob = self.Prob_Pos
            Neg_Prob = self.Prob_Neg

            if ("hash_trump" in tweets[i]) or ("hash_allahsoil" in tweets[i]) or ("hash_libtard" in tweets[i]) or ("hash_sjw" in tweets[i]):
                sentiment.append("1")
                decided = True
            elif ("hash_love" in tweets[i]) or ("hash_positive" in tweets[i]) or ("hash_smile" in tweets[i]):
                sentiment.append("0")
                decided = True


            if decided == False:

                for j in tweets[i]:
                    if j not in (stop):
                        word = self.Dict.get(j)

                        if word is not None:
                            update_pos = (word.get('P') + 1)/(self.Num_Pos + len(self.Dict))
                            update_neg = (word.get('N') + 1)/(self.Num_Neg + len(self.Dict))

                            Pos_Prob *= update_pos
                            Neg_Prob *= update_neg
                        else:
                            Pos_Prob *= 1
                            Neg_Prob *= 1

                if Neg_Prob >= Pos_Prob:
                    sentiment.append("1")
                else:
                    sentiment.append("0")

            decided = False

        with open("submit.csv", 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            for i in range(len(ids)):
                writer.writerow([ids[i], sentiment[i]])

