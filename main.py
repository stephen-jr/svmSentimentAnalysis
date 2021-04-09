from _utils import load_dataset, classifier, report_results, get_roc_curve, preprocesstexts, cnn_model

DATASET_NAME = ["Musical_Instruments_5", "Office_Products_5"]
# Get and Prepare Dataset
X_Train = X_Test = Y_Train = Y_Test = []
for dataset in DATASET_NAME:
    x_trn, x_tst, y_trn, y_tst = load_dataset('data/' + dataset + '.json')  # Do ensure you don't have an
    X_Train = [*X_Train, *x_trn]
    X_Test = [*X_Test, *x_tst]
    Y_Train = [*Y_Train, *y_trn]
    Y_Test = [*Y_Test, *y_tst]
                                                                                   # existing model with the same
                                                                                   # name. New model overwrites existing
                                                                                   # ones

# Create an SVC model fitted to the split dataset
svm_model = classifier(X_Train, X_Test, Y_Train, Y_Test, save_name='productReview', grid=False)  # grid=True Take longer but
                                                                                          # better model
# Model's Optimum Hyperparameters
print('SVM C parameter : ', svm_model.best_params_)
print("Model's Best Score", svm_model.best_score_)


# Model's Classification Report
# report = report_results(model.best_estimator_, X_Test, Y_Test)
report = report_results(svm_model, X_Test, Y_Test)
# Model's Confusion Matrix
print("Confusion Matrix := === === === ===")
print(report['confusion_matrix'])

# Model's Metric Analysis
print("Performance Metrics : === === ===")
print(f"Accuracy : {report['metrics']['acc']:{4}}")
print(f"Precision : {report['metrics']['precision']:{4}}")
print(f"Recall : {report['metrics']['recall']:{4}}")
print(f"F-Score : {report['metrics']['f1']:{4}}")
print(f"Area Under Curve(AUC) : {report['metrics']['auc']:{4}}")

# Plot ROC Curve
get_roc_curve(svm_model.best_estimator_, X_Test, Y_Test)

# Classify on a single Instance
svm_model.predict([preprocesstexts("flying with @united is always a great experience. If you don't lose your luggage")])

# CNN Feature Extractor
cnn_model(X_Train, X_Test, Y_Train, Y_Test, save_name='productReview')
