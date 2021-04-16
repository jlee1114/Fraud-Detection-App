from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib

def desc_tfidf_matrix(X):
    bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3], lowercase=False)
    tfidf_transform = TfidfTransformer(norm=None)
    X_bow = bow_transform.fit_transform(X['desc_cleaned'])
    X_tfidf = tfidf_transform.fit_transform(X_bow)
    return X_tfidf

def tfidf_prob(X,y):
    RMF = RandomForestClassifier().fit(X, y)
    y_prob = RMF.predict_proba(X)[:,1]
    return y_prob

def get_features(X,y):
    X_tfidf = desc_tfidf_matrix(X)
    y_prob = tfidf_prob(X_tfidf,y)
    X['tfidf_prob'] = y_prob
    X_features = X[['body_length','user_age','avg_costs','avg_quantity','tfidf_prob']]
    col_means = X_features.mean()
    X_features = X_features.fillna(col_means)
    return X_features

if __name__ == '__main__':
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    X_train_features = get_features(X_train,y_train)
    X_test_features = get_features(X_test,y_test)
    model = RandomForestClassifier().fit(X_train_features, y_train)
    y_pred = model.predict(X_test_features)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    joblib.dump(model, 'rmf.pkl')
    print('Precision Score with tf-idf', 'features', precision)
    print('Recall Score with tf-idf', 'features', recall)
    print('Confusion matrix with tf-idf', 'features', confusion_matrix(y_test, y_pred))

'''
Precision Score with tf-idf features 0.9506172839506173
Recall Score with tf-idf features 0.9361702127659575
Confusion matrix with tf-idf features
[[3240   16]
[  21  308]]
'''

