import os
import pickle
import pandas as pd
import logging
import argparse

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

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
        logging.info(f'Downloading {dataset} from Kaggle...')
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
        logging.error('Diabetes.csv not found.')
    except Exception as e:
        logging.error(f'Error while loading Diabetes.csv: {e}')

    diabetes_df = change_types(diabetes_df)
    logging.info(f'Data Changed to Integer')

    #diabetes_df = selecting_features(diabetes_df, data_dir)
    split_data(diabetes_df, data_dir)

def change_types(diabetes_df):
    for col in diabetes_df.columns:
        diabetes_df[col] = diabetes_df[col].astype(int)

    return diabetes_df

def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)
    logging.info(f'File {filename} created successfully.')

def selecting_features(diabetes_df, data_dir):
    # Top features for ML from Features.ipynb
    top_features = ['Diabetes_012', 'BMI', 'Age','Income','PhysHlth','Education','GenHlth','MentHlth','HighBP','Fruits']
    diabetes_df = diabetes_df[top_features].copy() 
    logging.info(f'Top features selected for dataset.')

    return diabetes_df

def balancing_classes(X_train, y_train):
    logging.info(f'Running class Re-balancing with SMOTETomek.....')
    smote = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f'Data re-balanced with SMOTETomek')

    return X_resampled, y_resampled


def split_data(diabetes_df, data_dir):
    X = diabetes_df.drop('Diabetes_012', axis=1)
    y = diabetes_df['Diabetes_012']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balancing_classes(X_train, y_train)
    logging.info(f'Data succesfully splitted to test and train (8/2)')

    dump_pickle((X_train, y_train), data_dir / 'train.pkl')
    dump_pickle((X_test, y_test), data_dir / 'test.pkl')


if __name__ == '__main__':
    args = parse_args()
    read_dataframe(force_download=args.force_download)


