import datetime as dt
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import mlflow
import pandas as pd
import psycopg2
import xgboost as xgb
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, RocAuc, ValueDrift
from evidently.presets import (
    ClassificationPreset,
    ClassificationQuality,
    DataDriftPreset,
    DataSummaryPreset,
)
from evidently.sdk.models import *
from evidently.sdk.panels import *
from evidently.tests import *
from evidently.ui.workspace import RemoteWorkspace
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
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

host = os.getenv("GRAFANA_DB_ENDPOINT")
CONNECTION_STRING = f"host={host} port=5432 user=grafana password=grafanagrafana"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=grafana"

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()
buffer = BytesIO()


def get_final_model():
    experiment_id = client.search_experiments()

    experiment_id = experiment_id[0].experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    run_name = runs[0].info.run_name
    run_id = runs[0].info.run_id
    run_artifacts_id = runs[0].info.artifact_uri

    logging.info(f"Final experiment found as {run_artifacts_id} with ID: {run_name}")

    bucket = "mlflow-bucket-diabetes"
    prefix = f"{experiment_id}/{run_id}/artifacts/model/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    files = [obj["Key"] for obj in response.get("Contents", [])]
    logging.info(f"Artifacts files: {files}")

    model_key = None
    for file_key in files:
        if file_key.endswith(".xgb"):
            model_key = file_key
            model_type = "xgboost"
            break
        if file_key.endswith(".cb"):
            model_key = file_key
            model_type = "catboost"
            break
        if file_key.endswith(".pkl"):
            model_key = file_key
            model_type = "sklearn"
            break

    logging.info(f"Model type is {model_type}")
    logging.info(f"Model key for S3 is {model_key}")

    try:
        model = s3.download_fileobj(Fileobj=buffer, Bucket=bucket, Key=model_key)
        buffer.seek(0)

        if model_type == "xgboost":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xgb") as tmp:
                tmp.write(buffer.read())
                tmp.flush()
                booster = xgb.Booster()
                booster.load_model(tmp.name)
                logging.info(f"Model: {run_id} found and dowloaded")
                return booster, model_type
        elif model_type == "catboost":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cb") as tmp:
                tmp.write(buffer.read())
                tmp.flush()
                model = CatBoostClassifier()
                model.load_model(tmp.name)
                return model, model_type
        elif model_type == "sklearn":
            model = joblib.load(buffer)
            return model
    except Exception as e:
        logging.error(f"No model was found/dowloaded. Details - {e}")
        return None


def load_and_prepare_data():
    input_dir = Path(__file__).resolve().parent.parent / "Data"

    try:
        X_ref = pd.read_parquet(input_dir / f"X_ref.parquet")
        y_ref = pd.read_parquet(input_dir / f"y_ref.parquet")
        X_curr = pd.read_parquet(input_dir / f"X_train.parquet")
        y_curr = pd.read_parquet(input_dir / f"y_train.parquet")
        logging.info("Reference and current data loaded.")
    except Exception as e:
        logging.error(f"Data not found in directory: {input_dir} error: {e}")

    return X_ref, y_ref, X_curr, y_curr


def load_data_from_s3(type):
    buffer = BytesIO()
    bucket = "diabetes-data-bucket"
    key = f"processed_data/{type}.parquet"

    s3.download_fileobj(Fileobj=buffer, Bucket=bucket, Key=key)
    buffer.seek(0)

    df = pd.read_parquet(buffer)

    if "y" in type.lower():
        if isinstance(df, pd.DataFrame):
            return df.squeeze().values.ravel()
        else:
            return df.ravel()
    else:
        return df


def load_reports_to_s3(report, report_name, type_of):
    bucket = "evidently-data-bucket"

    if type_of.lower().strip() == "classification":
        s3.upload_file(report, bucket, f"classification/{report_name}")
        logging.info(f"Classification report {report_name} uploaded to S3")
    elif type_of.lower().strip() == "prediction":
        s3.upload_file(report, bucket, f"predictions/{report_name}")
        logging.info(f"Predictions report {report_name} uploaded to S3")
    else:
        logging.error(f"Unknow type of report - {type_of}")


def model_predictions():

    X_ref = load_data_from_s3("X_ref")
    y_ref = load_data_from_s3("y_ref")
    X_curr = load_data_from_s3("X_train")
    y_curr = load_data_from_s3("y_train")
    model, model_type = get_final_model()

    if model_type == "xgboost":
        # Create predictions for referential data
        val = xgb.DMatrix(X_ref, label=y_ref)
        y_pred_proba = model.predict(val)
        X_ref["prediction_proba"] = y_pred_proba
        ref_data = X_ref.copy()
        ref_data["target"] = y_ref
        ref_data.to_parquet("reference_data.parquet")

        # Create predictions for current data
        train = xgb.DMatrix(X_curr, label=y_curr)
        y_pred_proba = model.predict(train)
        curr_data = X_curr.copy()
        curr_data["prediction_proba"] = y_pred_proba
        curr_data["target"] = y_curr
        return ref_data, curr_data

    elif model_type == "catboost":
        y_pred_proba = model.predict_proba(X_ref)[:, 1]
        X_ref["prediction_proba"] = y_pred_proba
        ref_data = X_ref.copy()
        ref_data["target"] = y_ref

        y_pred_proba = model.predict_proba(X_curr)[:, 1]
        curr_data = X_curr.copy()
        curr_data["prediction_proba"] = y_pred_proba
        curr_data["target"] = y_curr
        return ref_data, curr_data

    elif model_type == "randomforest":
        # Random Forest from sklearn
        y_pred_proba = model.predict_proba(X_ref)[:, 1]
        X_ref["prediction_proba"] = y_pred_proba
        ref_data = X_ref.copy()
        ref_data["target"] = y_ref
        y_pred_proba = model.predict_proba(X_curr)[:, 1]

        curr_data = X_curr.copy()
        curr_data["prediction_proba"] = y_pred_proba
        curr_data["target"] = y_curr
        return ref_data, curr_data

    elif model_type == "logisticregression":
        # Logistic Regression from sklearn
        y_pred_proba = model.predict_proba(X_ref)[:, 1]
        X_ref["prediction_proba"] = y_pred_proba
        ref_data = X_ref.copy()
        ref_data["target"] = y_ref

        y_pred_proba = model.predict_proba(X_curr)[:, 1]
        curr_data = X_curr.copy()
        curr_data["prediction_proba"] = y_pred_proba
        curr_data["target"] = y_curr
        return ref_data, curr_data
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def data_definition():
    ref_data, curr_data = model_predictions()

    num_cols = ["BMI", "prediction_proba"]
    cat_cols = [
        "Age",
        "Income",
        "PhysHlth",
        "Education",
        "GenHlth",
        "MentHlth",
        "HighBP",
    ]
    features = num_cols + cat_cols + ["target"]

    data_definiton = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
        classification=[
            BinaryClassification(target="target", prediction_labels="prediction_proba")
        ],
    )

    curr_dataset = Dataset.from_pandas(
        curr_data[features], data_definition=data_definiton
    )

    ref_dataset = Dataset.from_pandas(
        ref_data[features], data_definition=data_definiton
    )

    return ref_dataset, curr_dataset


def create_workspace(name):
    evidently_url = "http://evidently:8000"

    workspace = RemoteWorkspace(base_url=evidently_url)
    existing_projects = workspace.list_projects()
    description = "Monitoring for Diabetes project"

    for project in existing_projects:
        if project.name == name:
            logging.info(f"Workspace already exists: {name}")
            return project, workspace, 1

    logging.info(f"Creating new workspace: {name}")
    project = workspace.create_project(name, description)
    return project, workspace, 0


def creating_reports():

    ref_dataset, curr_dataset = data_definition()

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset(),
            ClassificationPreset(),
            ClassificationQuality(),
            RocAuc(),
        ],
        include_tests=True,
    )

    report_predictions = Report(
        metrics=[
            ValueDrift(column="prediction_proba"),
            DriftedColumnsCount(),
            MissingValueCount(column="prediction_proba"),
        ],
        include_tests=True,
    )

    snapshot_predictions = report_predictions.run(
        reference_data=ref_dataset, current_data=curr_dataset
    )
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)

    snapshot_predictions.save_html("report_predictions.html")
    snapshot.save_html("report.html")
    logging.info("Saved reports for classification and predictions summary.")

    name = "Diabetes project monitoring"
    project, workspace, status = create_workspace(name)

    today = dt.datetime.now()
    today = today.strftime("%d.%m.%Y-%H")

    name_report = f"Report for Classification model {today}.html"
    workspace.add_run(project.id, snapshot, name=name_report)
    logging.info(f"Created - Report for Classification model: {today}")
    load_reports_to_s3("report.html", name_report, "classification")

    name_report_predictions = f"Report for predictions statistics {today}.html"
    workspace.add_run(project.id, snapshot_predictions, name=name_report_predictions)
    logging.info(f"Created - Report for predictions statistics: {today}")
    load_reports_to_s3("report_predictions.html", name_report_predictions, "prediction")

    if status == 0:
        create_dashboard(project)

    insert_metrics_into_db(snapshot, snapshot_predictions)

    try:
        os.remove("report.html")
        os.remove("report_predictions.html")
        logging.info("Temporary report files deleted.")
    except Exception as e:
        logging.warning(f"Failed to delete temporary report files: {e}")


def create_dashboard(project):

    panel = project.dashboard.add_panel(text_panel(title="Diabetes Project Dashboard"))

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Number of Drifted Columns",
            subtitle="How many columns drifted from the training data.",
            size="full",
            values=[PanelMetric(legend="", metric="DriftedColumnsCount")],
            plot_params={"plot_type": "counter", "aggregation": "none"},
        ),
    )

    project.dashboard.add_panel(
        bar_plot_panel(
            title="Inference Count",
            values=[
                PanelMetric(
                    metric="RowCount",
                    legend="count",
                ),
            ],
            size="half",
        ),
    )

    project.dashboard.add_panel(
        line_plot_panel(
            title="Number of Missing Values",
            values=[
                PanelMetric(metric="DatasetMissingValueCount", legend="count"),
            ],
            size="half",
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="F1 Score",
            subtitle="F1 Score for current model.",
            size="half",
            values=[PanelMetric(legend="F1", metric="F1Score")],
            plot_params={"plot_type": "counter", "aggregation": "none"},
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Recall",
            subtitle="Recall for current model.",
            size="half",
            values=[PanelMetric(legend="Recall", metric="Recall")],
            plot_params={"plot_type": "counter", "aggregation": "none"},
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Accuracy",
            subtitle="Accuracy for current model.",
            size="half",
            values=[PanelMetric(legend="Accuracy", metric="Accuracy")],
            plot_params={"plot_type": "counter", "aggregation": "none"},
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="ROC-AUC",
            subtitle="Receiver Operating Characteristic (ROC) Curve",
            size="half",
            values=[PanelMetric(legend="ROC-AUC", metric="RocAuc")],
            plot_params={"plot_type": "counter", "aggregation": "none"},
        ),
    )

    project.save()
    logging.info("Dashboard with panels and counters created")


def prep_db():
    create_table_statement = """
    CREATE TABLE IF NOT EXISTS evidently_metrics (
        timestamp TIMESTAMP NOT NULL,
        prediction_drift FLOAT,
        num_drifted_columns INTEGER NOT NULL,
        accuracy FLOAT NOT NULL,
        precision FLOAT NOT NULL,
        recall FLOAT NOT NULL,
        f1_score FLOAT NOT NULL,
        roc_auc FLOAT NOT NULL
    );
    """

    with psycopg2.connect(CONNECTION_STRING) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname='grafana'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("CREATE DATABASE grafana;")

    with psycopg2.connect(CONNECTION_STRING_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_statement)

    conn.close()


def insert_metrics_into_db(snapshot, snapshot_predictions):
    prep_db()
    result = snapshot.dict()
    result_predictions = snapshot_predictions.dict()

    try:
        prediction_drift = result_predictions["metrics"][0]["value"]
        num_drifted_columns = result_predictions["metrics"][1]["value"]["count"]
        accuracy = result["metrics"][54]["value"]
        precision = result["metrics"][55]["value"]
        recall = result["metrics"][56]["value"]
        f1_score = result["metrics"][57]["value"]
        roc_auc = result["metrics"][65]["value"]
        logging.info("Given metrics were extraceted.")
    except (KeyError, IndexError) as e:
        logging.error(
            f"Error with extracting metrics: missing key/index in JSON structure - Error:{e}"
        )
    except Exception as e:
        logging.error(f"Error with getting data - Detail:{e}")

    report_timestamp = dt.datetime.now().replace(microsecond=0)
    logging.info(f"Timestamp for report is: {report_timestamp}")

    insert_statement = """
    INSERT INTO evidently_metrics (timestamp, prediction_drift, num_drifted_columns, accuracy, precision, recall, f1_score, roc_auc)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """

    with psycopg2.connect(CONNECTION_STRING_DB) as conn:
        conn.autocommit = True
        with conn.cursor() as curr:
            curr.execute(
                insert_statement,
                (
                    report_timestamp,
                    prediction_drift,
                    num_drifted_columns,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    roc_auc,
                ),
            )
            logging.info(f"Records inserted with timestamp: {report_timestamp}")


if __name__ == "__main__":
    logging.info("Starting monitoring....")

    try:
        prep_db()
        logging.info("Database is prepared")

        creating_reports()
        logging.info("Reports created and inserted into db")

    except Exception as e:
        logging.critical(f"Failure when monitoring: {e}", exc_info=True)
