
import os
import re
import mlflow
import pickle
import logging
import numpy as np
import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mlflow.set_tracking_uri('http://127.0.0.1:5000')

def load_data(filename):
    data_path = '../Data/'
    path = os.path.join(data_path, filename)
    with open(path, 'rb') as f_in:
        X, y  = pickle.load(f_in)

    if X.empty or y.empty:
        logging.error(f'{filename} data is empty')
    else:
        logging.info('Data Loaded succesfully')
    
    print(X.dtypes)
    print(X.head())

    return X, y

def data_selection():
    train = 'train.pkl'
    test = 'test.pkl'

    X_train, y_train = load_data(train)
    X_test, y_test = load_data(test)
    logging.info('Data test/train loaded succesfully.')

    print("\n--- X_train dtypes po načtení ---")
    print(X_train.dtypes)
    print("\n--- X_test dtypes po načtení ---")
    print(X_test.dtypes)

    return X_train, y_train, X_test, y_test

def create_experiment():
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    experiments_sorted = sorted(experiments, key=lambda x: x.creation_time, reverse=True)
    latest_experiment = experiments_sorted[0]

    try:
        if latest_experiment:
            mlflow.delete_experiment(latest_experiment.experiment_id)
            logging.info(f"Experiment '{latest_experiment}' (ID: {latest_experiment.experiment_id}) was archived.")
        else:
            logging.error(f"Experiment '{latest_experiment}' not found.")
    except Exception as e:
         logging.error(f"Error while archiving experiment: {e}")
    name = set_new_experiment_name()
    mlflow.set_experiment(name)

def set_new_experiment_name(base_name="classification_experiment"):
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

    matching = [e for e in experiments if e.name.startswith(base_name)]

    versions = []
    for e in matching:
        match = re.search(rf"{base_name}_v(\d+)", e.name)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    new_experiment_name = f"{base_name}_v{next_version}"

    return new_experiment_name

def catboost_objective(params):
    logging.info('Running Catboost model.....')
    X_train, y_train, X_test, y_test = data_selection()

    with mlflow.start_run():
        mlflow.set_tag('Model', 'Catboost')
        mlflow.log_params(params)

        categorical_features = ['Age', 'GenHlth', 'Education', 'Income']

        model = CatBoostClassifier(**params,
                                   cat_features=categorical_features,
                                   early_stopping_rounds=50,
                                   eval_metric='TotalF1')

        model.fit(X_train, y_train, 
                  eval_set=(X_test, y_test),
                  use_best_model=True, 
                  logging_level='Silent')

        y_pred = model.predict(X_test)

        # Logging important metrics
        score = f1_score(y_test, y_pred, average='macro')
        loss = 1 - score 
        mlflow.log_metric('f1_macro', score)
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

    return {'loss': loss, 'status': STATUS_OK}

def xgboost_objective(params):
    logging.info('Running XGBoost model.....')
    X_train, y_train, X_test, y_test = data_selection()

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)


    with mlflow.start_run():
        mlflow.set_tag('Model', 'XGboost')
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred_proba = booster.predict(valid) 
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        #Loggin important metrics
        cv_scores = cross_val_score(booster, X_train, y_train, cv=5, scoring='f1_macro')
        cv_score = cv_scores.mean()
        cv_std = cv_scores.std()
        mlflow.log_metric("cv_f1_mean", cv_score)
        mlflow.log_metric("cv_f1_std", cv_std)

        score = f1_score(y_test, y_pred, average='macro')
        loss = 1 - score
        mlflow.log_metric('f1_macro', score)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

    return {'loss': loss, 'status': STATUS_OK}

def rf_objective(params):
    logging.info('Running Random Forest model.....')
    X_train, y_train, X_test, y_test = data_selection()

    with mlflow.start_run():
        mlflow.set_tag('Model', 'Random Forest')
        mlflow.log_params(params)

        # Dependency bootstrap and oob_score
        if not params['bootstrap']:
            params['oob_score'] = False

        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=params['max_depth'],
            criterion=params['criterion'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            bootstrap=params['bootstrap'],
            oob_score=params['oob_score'],
            class_weight=params['class_weight'],
            random_state=42,
            n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        #Logging important metrics
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        cv_score = cv_scores.mean()
        cv_std = cv_scores.std()
        mlflow.log_metric("cv_f1_mean", cv_score)
        mlflow.log_metric("cv_f1_std", cv_std)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        loss = 1 - f1
        mlflow.log_metric('f1_macro', f1)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

        return {'loss': loss, 'status': STATUS_OK}
    
def logreg_objective(params):
    logging.info('Running LogReg model.....')
    X_train, y_train, X_test, y_test = data_selection()

    with mlflow.start_run():
        mlflow.set_tag('Model', 'LogisticRegression')
        mlflow.log_params(params)

        model = LogisticRegression(
            penalty=params['penalty'],
            C=params['C'],
            solver=params['solver'],
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #Loggin important metrics
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        cv_score = cv_scores.mean()
        cv_std = cv_scores.std()
        mlflow.log_metric("cv_f1_mean", cv_score)
        mlflow.log_metric("cv_f1_std", cv_std)

        score = f1_score(y_test, y_pred, average='macro')
        loss = 1 - score
        mlflow.log_metric('f1_macro', score)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

    return {'loss': loss, 'status': STATUS_OK}

def run_all_models():
    objectives = {
        'catboost': (catboost_objective, {
            'depth': scope.int(hp.quniform('depth', 4, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, -1.6),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', -2, 1),
            'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0),
            'random_strength': hp.uniform('random_strength', 0.0, 1.0),
            'border_count': scope.int(hp.quniform('border_count', 32, 64, 16)),
            'iterations': 500,
            'loss_function': 'Logloss',
            'verbose': 0,
            'random_seed': 42
        }),
        'xgboost': (xgboost_objective, {
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', -5, -1.6),
            'reg_alpha': hp.loguniform('reg_alpha', -8, 0),
            'reg_lambda': hp.loguniform('reg_lambda', -7, 0),
            'min_child_weight': hp.loguniform('min_child_weight', 1, 10),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'objective': 'binary:logistic',
            'num_class': 2,
            'seed': 42
        }),
        'random_forest': (rf_objective, {
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 100)),
            'max_depth': hp.choice('max_depth', [None] + [scope.int(hp.quniform('max_depth_val', 5, 50, 1))]),
            'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.4),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': hp.choice('max_leaf_nodes', [None] + [scope.int(hp.quniform('max_leaf_nodes_val', 10, 100, 1))]),
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'oob_score': hp.choice('oob_score', [True, False]),
            'class_weight': hp.choice('class_weight', ['balanced', None])
        }),
        'logistic_regression': (logreg_objective, {
            'penalty': hp.choice('penalty', ['l1', 'l2']),
            'C': hp.loguniform('C', -4, 2),
            'solver': hp.choice('solver', ['liblinear', 'saga'])
        })
    }

    for name, (fn, space) in objectives.items():
        logging.info(f"Running optimization for {name}")
        best_result = fmin(
            fn=fn,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
        )
        logging.info(f"Best result for {name}: {best_result}")

if __name__ == '__main__':
    run_all_models()