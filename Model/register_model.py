import logging
import os
from io import BytesIO

import boto3
import mlflow
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
mlflow.set_tracking_uri("http://mlflow:5000")
load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

s3 = boto3.client(
    service_name="s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)


def load_parquet_from_s3(type: str):
    buffer = BytesIO()
    bucket = "diabetes-data-bucket"
    key = f"processed_data/{type}.parquet"

    s3.download_fileobj(Fileobj=buffer, Bucket=bucket, Key=key)
    buffer.seek(0)

    df = pd.read_parquet(buffer)

    return df


def dict_change_dtypes(params):
    parsed_params = {}
    for k, v in params.items():
        try:
            parsed_val = int(v)
        except:
            try:
                parsed_val = float(v)
            except:
                parsed_val = v
        parsed_params[k] = parsed_val
    return parsed_params


def select_best_n_models():
    client = MlflowClient()
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    experiments_sorted = sorted(
        experiments, key=lambda x: x.creation_time, reverse=False
    )
    latest_experiment = experiments_sorted

    order = ["metrics.recall DESC", "metrics.f1_macro DESC", "metrics.accuracy DESC"]

    top_n = 5
    runs = client.search_runs(
        experiment_ids=latest_experiment[-1].experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=order,
    )

    return runs


def select_best_params():
    runs = select_best_n_models()

    best_run_id = runs[0].info.run_id
    logging.info(f"Best run_id is: {best_run_id}")

    best_run_name = runs[0].info.run_name
    logging.info(f"Best run name is: {best_run_name}")

    best_metrics = runs[0].data.metrics
    logging.info(f"Best metrics are: {best_metrics}")

    best_params = runs[0].data.params
    logging.info(f"Best parameters are: {best_params}")

    best_model = runs[0].data.tags["Model"]
    logging.info(f"Best model is: {best_model}")

    return best_params, best_model


def register_best_model():
    set_experiment_by_name()
    params, model = select_best_params()

    if model == "Catboost":
        catboost_train(params)
    if model == "XGboost":
        xgboost_train(params)
    if model == "Random Forest":
        random_forest_train(params)
    if model == "Logistic Regression":
        logistic_regression_train(params)

    logging.info(f"Best model {model} has been registered.")
    add_tags(model)


def catboost_train(params):
    X_train = load_parquet_from_s3(type="X_train")
    y_train = load_parquet_from_s3(type="y_train")
    X_test = load_parquet_from_s3(type="X_test")
    y_test = load_parquet_from_s3(type="y_test")

    categorical_features_indices = [
        X_train.columns.get_loc(col)
        for col in ["Age", "GenHlth", "Education", "Income"]
    ]

    params = dict_change_dtypes(params)
    with mlflow.start_run() as run:
        mlflow.set_tag("Model", "Catboost")
        mlflow.set_tag("Stage", "Final_model")
        mlflow.log_params(params)

        model = CatBoostClassifier(
            **params,
            cat_features=categorical_features_indices,
            early_stopping_rounds=50,
            eval_metric="TotalF1",
        )
        model.fit(
            X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, verbose=0
        )

        y_pred = model.predict(X_test)
        # Logging important metrics
        score = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("f1_macro", score)
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        recall = recall_score(y_test, y_pred, average="macro")
        mlflow.log_metric("recall", recall)

        mlflow.catboost.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="final-catboost-model")
        logging.info(f"Catboost model registered: {model_uri}")


def xgboost_train(params):
    X_train = load_parquet_from_s3(type="X_train")
    y_train = load_parquet_from_s3(type="y_train")
    X_test = load_parquet_from_s3(type="X_test")
    y_test = load_parquet_from_s3(type="y_test")

    params = dict_change_dtypes(params)
    with mlflow.start_run() as run:
        mlflow.set_tag("Model", "XGBoost")
        mlflow.set_tag("Stage", "Final_model")
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred_proba = booster.predict(dvalid)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        score = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("f1_macro", score)
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        recall = recall_score(y_test, y_pred, average="macro")
        mlflow.log_metric("recall", recall)

        mlflow.xgboost.log_model(booster, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="final-xgboost-model")
        mlflow.set_tag("ModelURI", model_uri)

        print(f"XGBoost model registered: {model_uri}")


def random_forest_train(params):
    X_train = load_parquet_from_s3(type="X_train")
    y_train = load_parquet_from_s3(type="y_train")
    X_test = load_parquet_from_s3(type="X_test")
    y_test = load_parquet_from_s3(type="y_test")

    params = dict_change_dtypes(params)
    with mlflow.start_run() as run:
        mlflow.set_tag("Model", "RandomForest")
        mlflow.set_tag("Stage", "Final_model")
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("f1_macro", score)
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        recall = recall_score(y_test, y_pred, average="macro")
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="final-randomforest-model")
        mlflow.set_tag("ModelURI", model_uri)

        logging.info(f"Random Forest model registered: {model_uri}")


def logistic_regression_train(params):
    X_train = load_parquet_from_s3(type="X_train")
    y_train = load_parquet_from_s3(type="y_train")
    X_test = load_parquet_from_s3(type="X_test")
    y_test = load_parquet_from_s3(type="y_test")

    params = dict_change_dtypes(params)
    with mlflow.start_run() as run:
        mlflow.set_tag("Model", "LogisticRegression")
        mlflow.set_tag("Stage", "Final_model")
        mlflow.log_params(params)

        model = LogisticRegression(**params, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("f1_macro", score)
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        recall = recall_score(y_test, y_pred, average="macro")
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="final-logreg-model")
        mlflow.set_tag("ModelURI", model_uri)

        print(f"Logistic Regression model registered: {model_uri}")


def add_tags(model):
    client = MlflowClient()
    if model == "Catboost":
        model_name = "final-catboost-model"
    if model == "XGboost":
        model_name = "final-xgboost-model"
    if model == "Random Forest":
        model_name = "final-randomforest-model"
    if model == "Logistic Regression":
        model_name = "final-logreg-model"

    versions = client.get_latest_versions(name=model_name)
    version = versions[0].version

    client.set_model_version_tag(model_name, version, "Validated_by", "QA")
    client.set_model_version_tag(model_name, version, "Stage", "Production")
    client.set_model_version_tag(model_name, version, "Created_by", "David")

    client.set_registered_model_alias(
        name=model_name, alias="Champion", version=version
    )

    champion_version = versions[0]

    old_champion = None
    for v in versions[1:]:
        if any(alias.name == "Champion" for alias in v.aliases):
            old_champion = v
            break

    client.set_registered_model_alias(
        model_name, alias="Champion", version=champion_version.version
    )
    logging.info(f"Alias 'Champion' set to version {champion_version.version}")

    if old_champion:
        client.delete_registered_model_alias(model_name, alias="Champion")
        logging.info(f"Alias 'Champion' removed from version {old_champion.version}")

        client.set_registered_model_alias(
            model_name, alias="Archived", version=old_champion.version
        )
        logging.info(f"Alias 'Archived' set to version {old_champion.version}")

    for v in versions[1:]:
        if old_champion and v.version == old_champion.version:
            continue
        client.delete_model_version(model_name, v.version)
        logging.info(f"Deleted version {v.version}")


def set_experiment_by_name():
    client = MlflowClient()
    experiments = client.search_experiments()
    partial_name = "classification_experiment"

    matching_experiments = sorted(
        [exp for exp in experiments if partial_name.lower() in exp.name.lower()],
        key=lambda x: x.creation_time,
        reverse=True,
    )

    if matching_experiments:
        mlflow.set_experiment(matching_experiments[0].name)
        logging.info(
            f'Experiment "{matching_experiments[0].name}" selected by partial match.'
        )
    else:
        logging.warning(f'No experiment matched partial name "{partial_name}".')


if __name__ == "__main__":
    register_best_model()
