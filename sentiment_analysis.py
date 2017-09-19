

import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)
def load_file():
    with open('review.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])

        return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(1, 1),binary='False',max_df = 0.5)
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data
    #return data

def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)


def learn_model_svm(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    # Perform classification with SVM, kernel=linear
    #classifier_linear = svm.LinearSVC()
    classifier_linear = svm.SVC(kernel='linear')
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data_train,target_train)
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    #t2 = time.time()
    evaluate_model(target_test,predicted)

def learn_model_logistic(data,target):
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    # Perform classification with SVM, kernel=linear
    #classifier_linear = svm.LinearSVC()
    classifier_linear = LogisticRegression()
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data_train,target_train)
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    #t2 = time.time()
    evaluate_model(target_test,predicted)
# read more about model evaluation metrics here
# http://scikit-learn.org/stable/modules/model_evaluation.html
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))

def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)
    learn_model_svm(tf_idf,target)
    learn_model_logistic(tf_idf,target)


main()

