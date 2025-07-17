import os
import logging
import pandas as pd
import boto3
import tempfile
import xgboost as xgb
from io import BytesIO
from catboost import CatBoostClassifier

from dotenv import load_dotenv
from flask import Flask, flash, render_template, request, redirect, url_for
logging.basicConfig(level=logging.INFO, format='%(asctime)s/%(levelname)s/%(message)s', force=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)

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

def prediction(predict_df):
    buffer = BytesIO()
    bucket = 'mlflow-bucket-diabetes'

    response = s3.list_objects_v2(Bucket=bucket)

    prefix = f'{experiment_id}/{run_id}/artifacts/model/'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    files = [obj['Key'] for obj in response.get('Contents', [])]
    logging.info(f'Artifacts files: {files}')

    model_key = None
    for file_key in files:
        if file_key.endswith('.xgb'):
            model_key = file_key
            model_type = 'xgboost'
            break
        elif file_key.endswith('.cb'):
            model_key = file_key
            model_type = 'catboost'
            break
        elif file_key.endswith('.pkl'):
            model_key = file_key
            model_type = 'sklearn'
            break

    logging.info(f'Model type is {model_type}')
    logging.info(f'Model key for S3 is {model_key}')

    try:
        model = s3.download_fileobj(Fileobj=buffer, Bucket=bucket, Key=model_key)
        buffer.seek(0)
        if model_type == 'xgboost':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xgb") as tmp:
                tmp.write(buffer.read())
                tmp.flush()
                booster = xgb.Booster()
                booster.load_model(tmp.name)
                logging.info(f'Model: {run_id} found and dowloaded')
                y_pred_proba = booster.predict(predict_df)
                result = (y_pred_proba >= 0.5).astype(int)
                return result
        elif model_type == 'catboost':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.cb') as tmp:
                tmp.write(buffer.read())
                tmp.flush()
                model = CatBoostClassifier()
                model.load_model(tmp.name)
                y_pred_proba = model.predict_proba(predict_df)[:, 1]
                result = (y_pred_proba >= 0.5).astype(int)
                return result
        elif model_type == 'sklearn':
            model = joblib.load(buffer)
            return 2
    except Exception as e:
        logging.error(f'No model was found/dowloaded. Details - {e}')
        return None


@app.route('/', methods=['POST', 'GET'])
def index():
    column_names_int = ['Age', 'Income', 'PhysHlth', 'Education', 'GenHlth', 'MentHlth', 'HighBP', 'Fruits']

    if request.method == 'POST':
        try:
            bmi_raw = float(request.form.get('BMI'))
            input_data = {
            'BMI': int(round(bmi_raw)),
            **{col: int(request.form.get(col)) for col in column_names_int if col != 'BMI'}
            }
            logging.info('Inputs properly posted')

            predict_df = pd.DataFrame([input_data])

            result = prediction(predict_df)

            return render_template('index.html', prediction_result = result)
        except (ValueError, TypeError):
            flash('Incorrect types of inputs')
            return redirect(url_for("index.html"))

    return render_template('index.html', prediction_result = None)

if __name__ == '__main__':
    app.run(debug=True)