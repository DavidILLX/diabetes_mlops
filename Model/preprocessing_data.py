import os
import pickle
import pandas as pd
import logging
import argparse

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing diabetes dataset')
    parser.add_argument('--force-download', 
                        action='store_true', 
                        help='Force download data from Kaggle')
    
    return parser.parse_args()

def read_dataframe(force_download=False):
    data_dir = Path(__file__).parent.parent / 'Data'
    os.makedirs(data_dir, exist_ok=True)

    dataset = 'alexteboul/diabetes-health-indicators-dataset'
    diabetes_path = 'diabetes_012_health_indicators_BRFSS2015.csv'
    csv_file = os.path.join(data_dir, diabetes_path)

    # Forcing dowloand if the data is not in the directory
    if force_download or not os.path.exists(csv_file):
        logging.info('Downloading data from Kaggle...')
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=data_dir, unzip=True)
    else:
        logging.info('Using existing dataset, skipping download.')

    try:
        diabetes_df = pd.read_csv(csv_file)
        logging.info(f'File .csv loaded. Num of rows: {len(diabetes_df)}')
    except FileNotFoundError:
        logging.info('Diabetes.csv not found.')
    except Exception as e:
        logging.info(f'Error while loading Diabetes.csv: {e}')

    diabetes_df = selecting_features(diabetes_df, data_dir)
    split_data(diabetes_df, data_dir)


def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)
    logging.info(f'File {filename} created successfully.')

def selecting_features(diabetes_df, data_dir):
    # Top features for ML from Features.ipynb
    top_features = ['Diabetes_012', 'BMI', 'Age','Income','PhysHlth','Education','GenHlth','MentHlth','HighBP','Fruits']
    diabetes_df = diabetes_df[top_features].copy() 

    # Scaling continous features
    continuous = ['Age', 'BMI', 'Income', 'PhysHlth', 'MentHlth']
    scaler = StandardScaler()
    diabetes_df.loc[:, continuous] = scaler.fit_transform(diabetes_df[continuous])

    dump_pickle(scaler, data_dir / 'scaler.pkl')

    return diabetes_df

def balancing_classes(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def split_data(diabetes_df, data_dir):
    X = diabetes_df.drop('Diabetes_012', axis=1)
    y = diabetes_df['Diabetes_012']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balancing_classes(X_train, y_train)

    dump_pickle((X_train, y_train), data_dir / 'train.pkl')
    dump_pickle((X_test, y_test), data_dir / 'test.pkl')


if __name__ == '__main__':
    args = parse_args()
    read_dataframe(force_download=args.force_download)


