from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import re
import sys
from textblob import Word

def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

data = pd.read_csv('dataset.csv')
x = data['news'].tolist()
y = data['type'].tolist()
y_names = list(set(y))

for index,value in enumerate(x):
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

X_train, X_test, Y_train, Y_test = train_test_split(x, np.array(y), test_size=0.20, random_state=42)


class news_classifier():
    
    def nb(self):
    #naivebayes
        
        count_vect = CountVectorizer(stop_words='english')
        X_train_counts = count_vect.fit_transform(X_train)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        
        clf = MultinomialNB().fit(X_train_tfidf, Y_train)
        
        docs_test = X_test
        X_test_counts = count_vect.transform(docs_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        
        predicted = clf.predict(X_test_tfidf)
        print("NaiveBayes", np.mean(predicted == Y_test))

        print metrics.classification_report(
            Y_test, predicted,
            target_names=y_names)
        
        #for doc, category in zip(docs_test, predicted):
        #    print '%r => %s' % (doc, twenty_train.target_names[category])
    
    def nbp(self):
        #pipeline
        text_clf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
        ])
        _ = text_clf.fit(X_train, Y_train)
        docs_test = X_test
        predicted = text_clf.predict(docs_test)
        print("NaiveBayes Using Pipline", np.mean(predicted == Y_test))

        print metrics.classification_report(
            Y_test, predicted,
            target_names=y_names)
        
        #for doc, category in zip(docs_test, predicted):
        #    print '%r => %s' % (doc, twenty_train.target_names[category])
        
    def sgd(self):
        #SGDClassifier
        text_clf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),
        ])
        _ = text_clf.fit(X_train, Y_train)
        docs_test = X_test
        predicted = text_clf.predict(docs_test)
        print("SGDClassifier", np.mean(predicted == Y_test))

        print metrics.classification_report(
            Y_test, predicted,
            target_names=y_names)
        
        #for doc, category in zip(docs_test, predicted):
        #    print '%r => %s' % (doc, twenty_train.target_names[category])
        
    def main(self, c):
        text_clf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),
        ])
        _ = text_clf.fit(X_train, Y_train)
        docs_test = c
        predicted = text_clf.predict(docs_test)

        for doc, category in zip(docs_test, predicted):
            print '%r => %s' % (doc, category)

    def test(self):
        self.nb()
        self.nbp()
        self.sgd()
    

if __name__=='__main__':

    k=news_classifier()

    if len(sys.argv) == 1:
        k.test()

    else:
        with open(sys.argv[1]) as f:
            content = f.readlines()

        content = [x.strip() for x in content]
        content = " ".join(content)                       

        k.main([content])
    


