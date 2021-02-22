from sklearn.ensemble import RandomForestClassifier
import optuna
import sklearn
from HW1 import databases
from HW1 import bagging_alg

# ---------------- random forest ------------------------ #

def basic_random_forest():
    db = databases.get_db("fake jobs")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    rand_forrest_model = RandomForestClassifier()
    rand_forrest_model.fit(X_train, y_train)
    bagging_alg.training_results(rand_forrest_model,X_train, y_train)
    bagging_alg.testing_results(rand_forrest_model,X_test, y_test)

def finding_random_forest_best(X_train,y_train):
    # check individuals
    best_n_estimators = (0, 0)
    best_min_samples_leaf = (0, 0)
    best_max_leaf_nodes = (0, 0)
    for i in range(2, 13):
        # n_estimators
        tree_model = RandomForestClassifier(n_estimators=i*10)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_n_estimators[1]:
            best_n_estimators = (i, accuracy)
        # min_samples_leaf
        tree_model = RandomForestClassifier(min_samples_leaf=i)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_min_samples_leaf[1]:
            best_min_samples_leaf = (i, accuracy)
        # max_leaf_nodes
        tree_model = RandomForestClassifier(max_leaf_nodes=i)
        accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
        if accuracy > best_max_leaf_nodes[1]:
            best_max_leaf_nodes = (i, accuracy)
        print(i)
    print("best_n_estimators:")
    print(best_n_estimators)
    print("best_min_samples_leaf:")
    print(best_min_samples_leaf)
    print("best_max_leaf_nodes:")
    print(best_max_leaf_nodes)

    # check combined of max_depth and min_samples_leaf
    best_combined = (0, 0, 0)
    for i in range(2, 13):
        for j in range(2, 13):
            tree_model = RandomForestClassifier(n_estimators=i * 10, min_samples_leaf=j)
            accuracy = bagging_alg.run_k_fold(tree_model,X_train,y_train)
            if accuracy > best_combined[2]:
                best_combined = (i, j, accuracy)
        print(i)
    print("best_combined: (n_estimators,min_samples_leaf,accuracy)")
    print(best_combined)

def objective(trial):
    db = databases.get_db('fake jobs')
    X_train = db.X_train
    y_train = db.y_train
    n_estimators = trial.suggest_int('n_estimators', 30, 100)
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
    min_samples_leaf = int(trial.suggest_loguniform('min_samples', 2, 30))
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_rf():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def random_forest_comparision():
    db = databases.get_db('fake jobs')
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # income: 77, 24, 5
    complex_forest_model = RandomForestClassifier(n_estimators=64, max_depth=28, min_samples_leaf=2)
    complex_forest_model.fit(X_train, y_train)
    print("basic:")
    basic_random_forest()
    print("complex")
    bagging_alg.training_results(complex_forest_model,X_train, y_train)
    bagging_alg.testing_results(complex_forest_model,X_test, y_test)
