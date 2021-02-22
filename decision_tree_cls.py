from sklearn.tree import DecisionTreeClassifier
import optuna
import sklearn
from HW1 import databases
from HW1 import bagging_alg

# ---------------- decision tree ------------------------ #
def basic_decision_tree():
    db = databases.get_db("fake jobs")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    bagging_alg.training_results(tree_model,X_train, y_train)
    bagging_alg.testing_results(tree_model,X_test, y_test)

def finding_decision_tree_best(X_train,y_train):
    # check individuals
    best_max_depth = (0, 0)
    best_min_samples_leaf = (0, 0)
    best_max_features = (0, 0)
    for i in range(1, 10):
        # max_depth
        tree_model = DecisionTreeClassifier(max_depth=i)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_max_depth[1]:
            best_max_depth = (i, accuracy)
        # min_samples_leaf
        tree_model = DecisionTreeClassifier(min_samples_leaf=i)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_min_samples_leaf[1]:
            best_min_samples_leaf = (i, accuracy)
        # max_features
        tree_model = DecisionTreeClassifier(max_features=i)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_max_features[1]:
            best_max_features = (i, accuracy)
    print("best_max_depth:")
    print(best_max_depth)
    print("best_min_samples_leaf:")
    print(best_min_samples_leaf)
    print("best_max_features:")
    print(best_max_features)

    # check combined of max_depth and min_samples_leaf
    best_combined = (0, 0, 0)
    for i in range(1, 10):
        for j in range(1, 10):
            tree_model = DecisionTreeClassifier(max_depth=i, min_samples_leaf=j)
            accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
            if accuracy > best_combined[2]:
                best_combined = (i, j, accuracy)
    print("best_combined: (max_depth,min_samples_leaf,accuracy)")
    print(best_combined)

def objective(trial):
    db = databases.get_db("fake jobs")
    X_train = db.X_train
    y_train = db.y_train
    max_features = int(trial.suggest_loguniform('max_features', 1, len(X_train.columns)))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
    min_samples_leaf = int(trial.suggest_loguniform('min_samples', 2, 30))
    clf = DecisionTreeClassifier(max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_dt():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def decision_tree_comparision():
    db = databases.get_db("fake jobs")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # income values - 12, 8, 9
    # fake jobs - 4, 21, 2
    complex_tree_model = DecisionTreeClassifier(max_features=5, max_depth=21, min_samples_leaf=2)
    complex_tree_model.fit(X_train, y_train)
    print("basic:")
    basic_decision_tree()
    print("complex")
    bagging_alg.training_results(complex_tree_model,X_train, y_train)
    bagging_alg.testing_results(complex_tree_model,X_test, y_test)