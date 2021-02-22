from rotation_forest import RotationForestClassifier
import sklearn
import optuna
from HW1 import bagging_alg
from HW1 import databases

# ------------------- rotation forest --------------------------- #
def basic_rotation_forest():
    db = databases.get_db("fake jobs")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    rotation_model = RotationForestClassifier()
    rotation_model.fit(X_train.to_numpy(), y_train)
    bagging_alg.training_results(rotation_model, X_train, y_train)
    bagging_alg.testing_results(rotation_model, X_test, y_test)

def objective(trial):
    db = databases.get_db('fake jobs')
    X_train = db.X_train
    y_train = db.y_train
    n_estimators = int(trial.suggest_loguniform('n_estimators', 1, 100))
    max_features = int(trial.suggest_loguniform('max_features', 1, len(X_train.columns)))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
    min_samples_leaf = int(trial.suggest_loguniform('min_samples', 2, 30))
    clf = RotationForestClassifier(n_estimators = n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features)
    return sklearn.model_selection.cross_val_score(clf, X_train.to_numpy(), y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_rf():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def rotation_forest_comparision():
    db = databases.get_db('fake jobs')
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # income: 14, 3, 13, 10
    complex_forest_model = RotationForestClassifier(n_estimators=51, max_features=3, max_depth=31, min_samples_leaf=3)
    complex_forest_model.fit(X_train, y_train)
    print("basic:")
    basic_rotation_forest()
    print("complex:")
    bagging_alg.training_results(complex_forest_model,X_train, y_train)
    bagging_alg.testing_results(complex_forest_model,X_test, y_test)
