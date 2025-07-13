import re
import mlflow
import logging
import boto3
import pandas as pd
import xgboost as xgb
import os

from dotenv import load_dotenv
from io import BytesIO
from pathlib import Path
from mlflow.tracking import MlflowClient
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mlflow.set_tracking_uri('http://mlflow:5000')
load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

s3 = boto3.client(
    service_name='s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def load_parquet_from_s3(type):
    buffer = BytesIO()
    bucket = 'diabetes-data-bucket'
    key = f'processed_data/{type}.parquet'
    
    s3.download_fileobj(Fileobj=buffer, Bucket=bucket, Key=key)
    buffer.seek(0)
    
    df = pd.read_parquet(buffer)
    return df

def load_parquet(prefix: str):
    input_dir = Path('/app/Data')

    X = pd.read_parquet(input_dir / f"X_{prefix}.parquet")
    y = pd.read_parquet(input_dir / f"y_{prefix}.parquet").squeeze()

    logging.info(f"Loaded X from {input_dir / f'X_{prefix}.parquet'}")
    logging.info(f"Loaded y from {input_dir / f'y_{prefix}.parquet'}")

    return X, y

def create_new_experiment():
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

def set_new_experiment_name():
    client = MlflowClient()
    base_name="classification_experiment"

    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    matching = [e for e in experiments if e.name.startswith(base_name)]

    versions = []
    for e in matching:
        match = re.search(rf"{base_name}_v(\d+)", e.name)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    new_experiment_name = f"{base_name}_v{next_version}"

    logging.info(f'Set up experiment {new_experiment_name}')
    return new_experiment_name

def catboost_objective(params, X_train, y_train, X_test, y_test):
    logging.info('Running Catboost model.....')

    with mlflow.start_run():
        mlflow.set_tag('Model', 'Catboost')
        mlflow.log_params(params)

        categorical_features_indices = [X_train.columns.get_loc(col) for col in ['Age', 'GenHlth', 'Education', 'Income']]


        model = CatBoostClassifier(**params,
                                   cat_features=categorical_features_indices,
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

def xgboost_objective(params, X_train, y_train, X_test, y_test):
    logging.info('Running XGBoost model.....')

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
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        #Loggin important metrics
        score = f1_score(y_test, y_pred, average='macro')
        loss = 1 - score
        mlflow.log_metric('f1_macro', score)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

    return {'loss': loss, 'status': STATUS_OK}

def rf_objective(params, X_train, y_train, X_test, y_test):
    logging.info('Running Random Forest model.....')

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
        f1 = f1_score(y_test, y_pred, average='macro')
        loss = 1 - f1
        mlflow.log_metric('f1_macro', f1)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        mlflow.log_metric('recall', recall)

        return {'loss': loss, 'status': STATUS_OK}
    
def logreg_objective(params, X_train, y_train, X_test, y_test):
    logging.info('Running LogReg model.....')

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
    X_train = load_parquet_from_s3('X_train') 
    y_train = load_parquet_from_s3('y_train') 
    X_test = load_parquet_from_s3('X_test') 
    y_test = load_parquet_from_s3('y_test') 

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
            fn=lambda params, fn=fn: fn(params, X_train, y_train, X_test, y_test),
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
        )
        logging.info(f"Best result for {name}: {best_result}")

if __name__ == '__main__':
    create_new_experiment()
    run_all_models()