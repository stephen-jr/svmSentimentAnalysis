from _utils import load_dataset, classifier, report_results, get_roc_curve, preprocesstexts

DATASET_NAME = "Musical_Instruments_5"
# Get and Prepare Dataset
X_Train, X_Test, Y_Train, Y_Test = load_dataset('data/' + DATASET_NAME + '.json')  # Do ensure you don't have an existing model with the same name. New model overwrites existing ones

# Create an SVC model fitted to the splitted dataset
model = classifier(X_Train, X_Test, Y_Train, Y_Test, save_name=DATASET_NAME, grid=False) # grid=True Take longer but better model

# Model's Optimum Hyperparameters
print('SVM C parameter : ', model.best_params_)
print("Model's Best Score", model.best_score_)

# Model's Classification Report
# report = report_results(model.best_estimator_, X_Test, Y_Test)
report = report_results(model, X_Test, Y_Test)
# Model's Confusion Matrix
print("Confusion Matix := === === === ===")
print(report['confusion_matrix'])

# Model's Metric Analyssis
print("Performace Metrics : === === ===")
print(f"Accuracy : {report['metrics']['acc']:{4}}")
print(f"Precision : {report['metrics']['precision']:{4}}")
print(f"Recall : {report['metrics']['recall']:{4}}")
print(f"F-Score : {report['metrics']['f1']:{4}}")
print(f"Area Under Curve(AUC) : {report['metrics']['auc']:{4}}")

# Plot ROC Curve
get_roc_curve(model.best_estimator_, X_Test, Y_Test)

# Classify on a single Instance
model.predict([preprocesstexts("flying with @united is always a great experience. If you don't lose your luggage")])
