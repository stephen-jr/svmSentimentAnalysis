from imports import *


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


def tokenize_texts(x_trn, x_tst):
    maxlen = 70
    tokenizer = Tokenizer(10000)
    tokenizer.fit_on_texts(x_trn)
    X_train = tokenizer.texts_to_sequences(x_trn)
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test = tokenizer.texts_to_sequences(x_tst)
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    return X_train, X_test, maxlen, vocab_size


def define_model(vcb_size, mxln, mtrcs):
    model = Sequential()
    model.add(Embedding(vcb_size, 200, input_length=mxln))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=mtrcs)
    model.summary()
    return model


def cnn_feature_extractor(model, x_train, x_test):
    for layer in model.layers:
        print(layer)
    feature_extractor = Model(model.layers[0].input, outputs=[model.layers[7].output])
    x_train_features = feature_extractor(x_train)
    x_test_features = feature_extractor(x_test)
    x_train_features = np.array(x_train_features)
    x_test_features = np.array(x_test_features)
    return x_train_features, x_test_features


def train_model(x_train, x_test, y_train, y_test, model=None):
    assert model is not None
    es = EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1)
    tfb = TensorBoard(log_dir="./logs/cnn", histogram_freq=1)
    start = time()
    history = model.fit(x_train,
                        y_train,
                        epochs=30,
                        verbose=True,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        callbacks=[es, tfb])
    training_time = time() - start
    print("Training Time : ", training_time)

    _, tp, tn, fp, fn, acc, pre, rec, auc = model.evaluate(x_test, y_test)
    print("Accuracy : ", acc)
    print("Precision : ", pre)
    print("Recall : ", rec)
    print("AUC : ", auc)
    return history, model


def classifier(x_train, x_test, y_train, y_test, save_name=None, grid=None):
    print("=== === === Building SVM Classifier === === ===")
    svm = None
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))
    if grid:
        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        grid_svm = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.1, 1]},
                                cv=kfolds,
                                scoring="roc_auc",
                                verbose=1,
                                n_jobs=-1)

        grid_svm.fit(x_train, y_train)
        svm = grid_svm
    else:
        svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))
        svm.fit(x_train, y_train)
    print("\n=========================================================")
    try:
        print("Model's Validation Score : ", svm.score(x_test, y_test))
    except AttributeError:
        pass
    print("==========================================================")
    print("=== === === Build Complete === === ===")
    print("Saving Model ----------->")
    if save_name is not None:
        path = 'saves/' + save_name + '.pkl'
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'wb') as f:
            dill.dump(svm, f)
    print("Save Complete ---------->")
    return svm


def cnn_model(x_train, x_test, y_train, y_test, save_name=None):
    METRICS = [
        TruePositives(name='tp'),
        TrueNegatives(name='tn'),
        FalsePositives(name='fp'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc'),
    ]

    x_train, x_test, maxlen, vocab_size = tokenize_texts(x_train, x_test)
    model = define_model(vocab_size, maxlen, METRICS)
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    history, model = train_model(x_train, x_test, y_train, y_test, model)
    if save_name is not None:
        model.save('saves/'+save_name)

    x_train_features, x_test_features = cnn_feature_extractor(model, x_train, x_test)
    parameters = {'kernel': ['rbf'],
                  'C': [1, 10, 100, 1000],
                  'gamma': [1e-3, 1e-4]}
    clf = GridSearchCV(SVC(), parameters)
    clf.fit(x_train_features, y_train)
    svmclf = clf.best_estimator_
    svmclf.fit(x_train_features, y_train)
    y_pred_svm = svmclf.predict(x_test_features)
    svm_report = classification_report(y_test, y_pred_svm)
    print(svm_report)


def report_results(model, x, y):
    auc = None
    if isinstance(model, SVC):
        pred_proba = model.predict_proba(x)[:, 1]
        auc = roc_auc_score(y, pred_proba)
    pred = model.predict(x)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    if isinstance(model, SVC):
        result['auc'] = auc
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
