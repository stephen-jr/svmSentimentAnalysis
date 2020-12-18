import os
import re
import dill
import nltk
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score, f1_score, roc_curve


def replace_parenth(arr):
    return [txt.replace(')', '[)}]]').replace('(', '[({[]') for txt in arr]


def regex_join(arr):
    return '(' + '|'.join(arr) + ')'


def stem(txt):
    stemmer = nltk.stem.PorterStemmer()
    _stem = ''
    words = [word if (word[0:2] == '__') else word.lower() for word in txt.split() if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    _stem = ' '.join(words)
    return _stem


def preprocesstexts(text):
    # Handling Emoticons
    emoticons = [('__positive__',
                  [':-)', ':)', '(:', '(-:', ':-D', ':D', 'X-D', 'XD', 'xD', '<3', ':\\*', ';-)', ';)', ';-D', ';D',
                   '(;',
                   '(-;', ]), ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ])]

    text = re.sub('((www.[^\s]+)|(https?://[^\s]+))', '_URL_', text)  # Convert links to token '_URL_'
    text = re.sub('@[^\s]+', '_HANDLE_', text)  # Convert @username(handles) to token __HANDLE__
    text = re.sub(r'#([^\s]+)', r'\1', text)  # Replace #word with word
    text = text.strip('\'"')  # Trim text
    rpt_regex = re.compile(r"(.)\1+", re.IGNORECASE)
    text = rpt_regex.sub(r"\1\1", text)

    emoticons_regex = [(repl, re.compile(regex_join(replace_parenth(regx)))) for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        text = re.sub(regx, ' ' + repl + ' ', text)

    # text = text.lower()  # Convert to lower case

    text = stem(text)
    return text


def load_dataset(path):
    print("=== Loading Dataset ===")
    f_xtn = path.split('.')[-1]
    f = None
    if f_xtn == 'csv':
        f = pd.read_csv(path, error_bad_lines=False, encoding="ISO-8859-1")
    elif f_xtn == 'json':
        f = pd.read_json(path, lines=True)
    else:
        exit("Invalid File Extension")
    sentences = f['reviewText']
    rating = f['overall'].astype('int')
    print("\t----- Preprocessing Texts -------")
    prepocessed_text = sentences.apply(preprocesstexts)
    rating = [1 if x >= 3 else 0 for x in rating]
    print("\t------- Preprocessing Complete ---------")

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(prepocessed_text, rating, test_size=0.3, random_state=42)
    print("=== Dataset Load Complete ===")
    return X_Train, X_Test, Y_Train, Y_Test


def classifier(x_train, x_test, y_train, y_test, save_name=None):
    print("=== === === Building SVM Classifier === === ===")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc",
                            verbose=1,
                            n_jobs=-1)

    grid_svm.fit(x_train, y_train)
    print("\n=========================================================")
    print("Model's Validation Score : ", grid_svm.score(x_test, y_test))
    print("==========================================================")
    print("=== === === Build Complete === === ===")
    print("Saving Model ----------->")
    if save_name is not None:
        path = 'saves/' + save_name + '.pkl'
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as f:
            dill.dump(grid_svm, f)
    print("Save Complete ---------->")
    return grid_svm


def report_results(model, x, y):
    pred_proba = model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    c_matrix = confusion_matrix(y, pred)
    return {'metrics': result, 'confusion_matrix': c_matrix}


def get_roc_curve(model, x, y):
    pred_proba = model.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.show()
