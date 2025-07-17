import os
import logging
import pandas as pd
import boto3
import tempfile
import xgboost as xgb
from io import BytesIO

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
    bucket = 'mlflowmlflow-bucket-diabetes'

    response = s3.list_objects_v2(Bucket=bucket)

    if 'Contents' not in response:
        raise FileNotFoundError('No model artifacts found.')
    
    sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)

    latest_obj = sorted_objects[0]
    key = latest_obj['Key']
    logging.info(f'Latest model found: {key}')


    try:
        s3.download_fileobj(buffer, bucket, key)
        buffer.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xgb") as tmp:
            tmp.write(buffer.read())
            tmp.flush()
            booster = xgb.Booster()
            booster.load_model(tmp.name)
            pred_proba = booster.predict(predict_df)
            result = int(pred_proba[0] > 0.5)
            logging.info(f'Model found and made prediction: Probability - {pred_proba}, Classified - {result}')
        return result
    except Exception as e:
        logging.error(f'No model was found. Details - {e}')
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

            return render_template('index.html', results = result)
        except (ValueError, TypeError):
            flash('incorrect types of inputs')
            return redirect(url_for("index"))

    return render_template('index.html', results = None)

if __name__ == '__main__':
    app.run(debug=True)