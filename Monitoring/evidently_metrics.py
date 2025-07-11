import datetime as dt
import logging 
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
import psycopg2

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.ui.workspace import RemoteWorkspace
from evidently.sdk.panels import *
from evidently.sdk.models import *
from evidently.legacy.renderers.html_widgets import WidgetSize
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount, RocAuc
from evidently.presets import DataSummaryPreset, DataDriftPreset, ClassificationPreset, ClassificationQuality
from evidently.tests import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

CONNECTION_STRING = "host=localhost port=5432 user=grafana password=grafana"
CONNECTION_STRING_DB = CONNECTION_STRING + "dbname=grafana"

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()

def get_final_model():
    experiment_id = client.search_experiments()

    experiment_id = experiment_id[0].experiment_id

    runs = client.search_runs(experiment_ids=[experiment_id],
                              order_by=["attributes.start_time DESC"],
                              max_results=1,
                              )

    run_name = runs[0].info.run_name
    run_id = runs[0].info.run_id
    run_artifacts_id = runs[0].info.artifact_uri
    model_uri =runs[0].data.tags['ModelURI']

    logging.info(f'Final experiment found at: {model_uri} as {run_name} with ID: {run_artifacts_id}')

    local_path = client.download_artifacts(run_id, "model", dst_path='../artifacts/')

    local_path = "../artifacts/model/"
    try:
        model = mlflow.xgboost.load_model(local_path)
        logging.info('Model artifact found and loaded')
    except Exception as e:
        logging.error(f'Error when looking for model: {e}')

    return model

def load_and_prepare_data():
    input_dir = Path(__file__).resolve().parent.parent / 'Data'

    try:
        X_ref = pd.read_parquet(input_dir / f"X_ref.parquet")
        y_ref = pd.read_parquet(input_dir / f"y_ref.parquet")
        X_curr = pd.read_parquet(input_dir / f"X_train.parquet")
        y_curr = pd.read_parquet(input_dir / f"y_train.parquet")
        logging.info('Reference and current data loaded.')
    except Exception as e:
        logging.error(f'Data not found in directory: {input_dir}')

    return X_ref, y_ref, X_curr, y_curr

def model_predictions():  

    X_ref, y_ref, X_curr, y_curr = load_and_prepare_data()
    model = get_final_model()

    # Create predictions for referential data
    val = xgb.DMatrix(X_ref, label=y_ref)
    y_pred_proba = model.predict(val)
    X_ref['prediction_proba'] = y_pred_proba
    ref_data = X_ref.copy()
    ref_data['target'] = y_ref

    ref_data.to_parquet('reference_data.parquet')

    # Create predictions for current data
    train = xgb.DMatrix(X_curr, label=y_curr)
    y_pred_proba = model.predict(train)
    curr_data = X_curr.copy()
    curr_data['prediction_proba'] = y_pred_proba
    curr_data['target'] = y_curr

    return ref_data, curr_data

def data_definition():
    ref_data, curr_data = model_predictions()

    num_cols = ['BMI', 'prediction_proba']
    cat_cols = ['Age', 'Income', 'PhysHlth', 'Education', 'GenHlth', 'MentHlth', 'HighBP']
    features = num_cols + cat_cols + ['target', 'prediction']

    data_definiton = DataDefinition(numerical_columns=num_cols,
                                    categorical_columns=cat_cols,                                
                                    classification=[BinaryClassification(
                                            target='target',
                                            prediction_labels='prediction_proba')
                                            ])

    curr_dataset = Dataset.from_pandas(
        curr_data[features],
        data_definition=data_definiton
    )

    ref_dataset = Dataset.from_pandas(
        ref_data[features],
        data_definition=data_definiton
    )

    return ref_dataset, curr_dataset

def create_workspace(name):
    workspace = RemoteWorkspace(base_url='http://localhost:8000')
    workspace = create_workspace()

    description = 'Monitoring for Diabetes project'
    project = workspace.create_project(name, description)

    return project, workspace

def creating_reports():

    ref_dataset, curr_dataset = data_definition()

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
        ClassificationPreset(),
        ClassificationQuality(),
        RocAuc()
    ], include_tests=True)

    report_predictions = Report(metrics=[
        ValueDrift(column='prediction_proba'),
        DriftedColumnsCount(),
        MissingValueCount(column='prediction_proba')
    ], include_tests=True)

    snapshot_predictions = report_predictions.run(reference_data=ref_dataset, current_data=curr_dataset)
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)

    snapshot_predictions.save_html('report_predictions.html')
    snapshot.save_html('report.html')
    logging.info('Saved reports for classification and predictions summary.')

    name = 'Diabetes project monitoring'
    project, workspace = create_workspace(name)
    logging.info(f'Workspace created for {name}.')

    today = dt.datetime.now()
    today = today.strftime('%d.%m.%Y-%H')

    name_report = f'Report for Classification model: {today}'
    workspace.add_run(project.id, snapshot, name=name_report)
    logging.info(f'Created - Report for Classification model: {today}')

    name_report_predictions = f'Report for predictions statistics: {today}'
    workspace.add_run(project.id, snapshot_predictions, name=name_report_predictions)
    logging.info(f'Created - Report for predictions statistics: {today}')

    create_dashboard(project)
    insert_metrics_into_db(snapshot, snapshot_predictions)

def create_dashboard(project):

    panel = project.dashboard.add_panel(
        text_panel(title='Diabetes Project Dashboard')
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Number of Drifted Columns",
            subtitle="How many columns drifted from the training data.",
            size="full",
            values=[PanelMetric(legend="", metric="DriftedColumnsCount")],

            plot_params={"plot_type": "counter", "aggregation": "none"}
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
                PanelMetric(
                    metric="DatasetMissingValueCount",
                    legend="count"
                ),
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
            plot_params={"plot_type": "counter", "aggregation": "none"}
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Recall",
            subtitle="Recall for current model.",
            size="half",
            values=[PanelMetric(legend="Recall", metric="Recall")],
            plot_params={"plot_type": "counter", "aggregation": "none"}
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Accuracy",
            subtitle="Accuracy for current model.",
            size="half",
            values=[PanelMetric(legend="Accuracy", metric="Accuracy")],

            plot_params={"plot_type": "counter", "aggregation": "none"}
        ),
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="ROC-AUC",
            subtitle="Receiver Operating Characteristic (ROC) Curve",
            size="half",
            values=[PanelMetric(legend="ROC-AUC", metric="RocAuc")],

            plot_params={"plot_type": "counter", "aggregation": "none"}
        ),
    )

    project.save()
    logging.info('Dashboard with panels and counters created')

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
        conn.autocommit=True
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='grafana'")
        if len(res.fetchall()) == 0:
                conn.execute("CREATE DATABASE grafana;")
        with psycopg2.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(create_table_statement)
        
    conn.close()

def insert_metrics_into_db(snapshot, snapshot_predictions):
    prep_db()
    result = snapshot.dict()
    result_predictions = snapshot_predictions.dict()

    try:
        prediction_drift = result_predictions['metrics'][0]['value']
        num_drifted_columns = result_predictions['metrics'][1]['value']['count']
        accuracy = result['metrics'][54]['value']
        precision = result['metrics'][55]['value']
        recall = result['metrics'][56]['value']
        f1_score = result['metrics'][57]['value']
        roc_auc = result['metrics'][65]['value']
        logging.info('Given metrics were extraceted.')
    except (KeyError, IndexError) as e:
            logging.error(f'Error with extracting metrics: missing key/index in JSON structure - Error:{e}')
    except Exception as e:
        logging.error(f'Error with getting data - Detail:{e}')

    report_timestamp = dt.datetime.now().replace(microsecond=0)
    logging.info(f'Timestamp for report is: {report_timestamp}')

    insert_statement = """
    INSERT INTO evidently_metrics (timestamp, prediction_drift, num_drifted_columns, accuracy, precision, recall, f1_score, roc_auc)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
            
    with psycopg2.connect(CONNECTION_STRING_DB) as conn:
        conn.autocommit = True
        with conn.cursor() as curr:
              curr.execute(insert_statement,(
                report_timestamp,
                prediction_drift,
                num_drifted_columns,
                accuracy,
                precision,
                recall,
                f1_score,
                roc_auc)) 
              logging.info(f'Recors inserted with timestamp:{report_timestamp}')


if __name__ == '__main__':
    logging.info('Starting monitoring....')
    
    try:
        prep_db() 
        logging.info('Database is prepared')

        creating_reports()
        logging.info('Reports created and inserted into db')

    except Exception as e:
        logging.critical(f'Failure when monitoring: {e}', exc_info=True)