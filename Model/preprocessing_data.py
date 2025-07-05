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
    test_data = 'diabetes_012_health_indicators_BRFSS2015.csv'
    train_data = 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

    test_csv_file = os.path.join(data_dir, test_data)
    train_csv_file = os.path.join(data_dir, train_data)

    # Forcing dowloand if the data is not in the directory
    if force_download or not os.path.exists(train_csv_file):
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
        train_df = pd.read_csv(train_csv_file)
        logging.info(f'File .csv loaded for train data. Num of rows: {len(train_df)}')

        test_df = pd.read_csv(test_csv_file)
        logging.info(f'File .csv loaded for train data. Num of rows: {len(test_df)}')
    except FileNotFoundError:
        logging.error('Diabetes.csv not found.')
    except Exception as e:
        logging.error(f'Error while loading Diabetes.csv: {e}')

    test_df = change_types(test_df)
    train_df = change_types(train_df)
    logging.info(f'Data Changed to Integer')

    train_df = selecting_features(train_df)
    test_df = selecting_features(test_df)

    # Changing clases from pre-diabetes to diabetes
    test_df = change_classes(test_df)

    split_data(train_df, data_dir)
    split_data(test_df,data_dir)

def change_types(df):
    for col in df.columns:
        df[col] = df[col].astype(int)

    return df

def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)
    logging.info(f'File {filename} created successfully.')

def selecting_features(df):
    possible_targets = ['Diabetes_012', 'Diabetes_binary']
    target_col = next((col for col in df.columns if col in possible_targets), None)

    # Top features for ML from Features.ipynb
    other_features = ['BMI', 'Age', 'Income', 'PhysHlth', 'Education', 'GenHlth', 'MentHlth', 'HighBP', 'Fruits']
    selected_cols = [target_col] + other_features

    df = df[selected_cols].copy()
    logging.info(f"Top features selected for dataset. Target: {target_col}")

    return df

def balancing_classes(X_train, y_train):
    logging.info(f'Running class Re-balancing with SMOTETomek.....')
    smote = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f'Data re-balanced with SMOTETomek')

    return X_resampled, y_resampled

def split_data(df, data_dir):
    if df.columns[0] == 'Diabetes_012':
        X_test = df.drop('Diabetes_012', axis=1)
        y_test = df['Diabetes_012']
        dump_pickle((X_test, y_test), data_dir / 'test.pkl')
        logging.info(f'Data succesfully splitted to test')
    elif df.columns[0] == 'Diabetes_binary':
        X_train = df.drop('Diabetes_binary', axis=1)
        y_train = df['Diabetes_binary']
        dump_pickle((X_train, y_train), data_dir / 'train.pkl')
        logging.info(f'Data succesfully splitted to train')
    else:
        logging.error('Unknown target column -> data was not saved.')

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train, y_train = balancing_classes(X_train, y_train)
    #logging.info(f'Data succesfully splitted to test and train (8/2)')

def change_classes(test_df):
    test_df = test_df.replace(2, 1)

    return test_df

if __name__ == '__main__':
    args = parse_args()
    read_dataframe(force_download=args.force_download)


