import argparse
import logging
import os
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv
from imblearn.combine import SMOTETomek
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing diabetes dataset")
    parser.add_argument(
        "--force-download", action="store_true", help="Force download data from Kaggle"
    )

    return parser.parse_args()


def read_dataframe(force_download=False):
    os.makedirs("/home/airflow/.config/kaggle", exist_ok=True)
    data_dir = Path(__file__).resolve().parent.parent / "Data"
    os.makedirs(data_dir, exist_ok=True)

    dataset = "alexteboul/diabetes-health-indicators-dataset"
    test_data = "diabetes_012_health_indicators_BRFSS2015.csv"
    train_data = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    ref_data = "diabetes_binary_health_indicators_BRFSS2015.csv"

    test_csv_file = os.path.join(data_dir, test_data)
    train_csv_file = os.path.join(data_dir, train_data)
    ref_csv_file = os.path.join(data_dir, ref_data)

    # Forcing dowloand if the data is not in the directory
    if force_download or not os.path.exists(train_csv_file):
        logging.info(f"Downloading {dataset} from Kaggle...")
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=data_dir, unzip=True)
    else:
        logging.info("Using existing dataset, skipping download.")

    upload_to_s3(test_csv_file, "test_data")
    upload_to_s3(train_csv_file, "train_data")
    upload_to_s3(ref_csv_file, "ref_data")

    try:
        train_df = pd.read_csv(train_csv_file)
        logging.info(f"File .csv loaded for train data. Num of rows: {len(train_df)}")

        test_df = pd.read_csv(test_csv_file)
        logging.info(f"File .csv loaded for train data. Num of rows: {len(test_df)}")

        ref_df = pd.read_csv(ref_csv_file)
        logging.info(f"File .csv loaded for ref data. Num of rows: {len(ref_df)}")
    except FileNotFoundError:
        logging.error("Diabetes files not found.")
    except Exception as e:
        logging.error(f"Error while loading Diabetes files: {e}")

    test_df = change_types(test_df)
    train_df = change_types(train_df)
    ref_df = change_types(ref_df)
    logging.info(f"Data Changed to Integer")

    train_df = selecting_features(train_df)
    test_df = selecting_features(test_df)
    ref_df = selecting_features(ref_df)

    # Changing clases from pre-diabetes to diabetes
    test_df = change_classes(test_df)

    split_data(train_df, prefix="train")
    split_data(test_df, prefix="test")
    split_data(ref_df, prefix="ref")


def upload_to_s3(input, name):
    aws_bucket = "diabetes-data-bucket"
    buffer = BytesIO()

    try:
        if isinstance(input, (pd.DataFrame, pd.Series)):
            df = input

            if isinstance(df, pd.Series):
                df = df.to_frame()

            df.to_parquet(buffer, engine="pyarrow", index=False)
            processed_folder = f"processed_data/{name}.parquet"

            try:
                buffer.seek(0)
                s3.upload_fileobj(
                    Fileobj=buffer, Bucket=aws_bucket, Key=processed_folder
                )
                logging.info(f"Processed file {name} uploaded to S3 successfully.")
            except Exception as e:
                logging.error(
                    f"Failed to upload preprocessed files to S3 details - {e}"
                )

        elif isinstance(input, str) and input.endswith(".csv"):
            if not os.path.exists(input):
                raise FileNotFoundError(f"CSV file not found: {input}")
            df = pd.read_csv(input)
            df.to_parquet(buffer, engine="pyarrow", index=False)
            raw_folder = f"raw_data/{name}.parquet"

            try:
                buffer.seek(0)
                s3.upload_fileobj(Fileobj=buffer, Bucket=aws_bucket, Key=raw_folder)
                logging.info(f"File {name} uploaded to S3 successfully.")
            except Exception as e:
                logging.error(f"Failed to upload files to S3 details - {e}")

        else:
            raise ValueError("Input must be a pandas DataFrame or a path to a CSV file")
    except Exception as e:
        logging.error(f"Failed to save Parquet to S3: {str(e)}")
        raise


def change_types(df):
    for col in df.columns:
        df[col] = df[col].astype(int)

    return df


def save_parquet(X, y, prefix, output_dir):
    output_dir = Path(output_dir)
    X_path = output_dir / f"X_{prefix}.parquet"
    y_path = output_dir / f"y_{prefix}.parquet"

    X.to_parquet(X_path, index=False)
    y.to_frame().to_parquet(y_path, index=False)

    logging.info(f"Saved X to {X_path}")
    logging.info(f"Saved y to {y_path}")


def selecting_features(df):
    possible_targets = ["Diabetes_012", "Diabetes_binary"]
    target_col = next((col for col in df.columns if col in possible_targets), None)

    # Top features for ML from Features.ipynb
    other_features = [
        "BMI",
        "Age",
        "Income",
        "PhysHlth",
        "Education",
        "GenHlth",
        "MentHlth",
        "HighBP",
        "Fruits",
    ]
    selected_cols = [target_col] + other_features

    df = df[selected_cols].copy()
    logging.info(f"Top features selected for dataset. Target: {target_col}")

    return df


def balancing_classes(X_train, y_train):
    logging.info(f"Running class Re-balancing with SMOTETomek.....")
    smote = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Data re-balanced with SMOTETomek")

    return X_resampled, y_resampled


def split_data(df, prefix):
    if prefix == "test":
        X_test = df.drop("Diabetes_012", axis=1)
        y_test = df["Diabetes_012"]

        # save_parquet(X_test, y_test, prefix="test", output_dir=data_dir)
        logging.info(f"Data succesfully splitted to test X,y")
        upload_to_s3(X_test, "X_test")
        upload_to_s3(y_test, "y_test")
    elif prefix == "train":
        X_train = df.drop("Diabetes_binary", axis=1)
        y_train = df["Diabetes_binary"]

        # save_parquet(X_train, y_train,  prefix="train", output_dir=data_dir)
        logging.info(f"Data succesfully splitted to train X,y")
        upload_to_s3(X_train, "X_train")
        upload_to_s3(y_train, "y_train")

    elif prefix == "ref":
        X_ref = df.drop("Diabetes_binary", axis=1)
        y_ref = df["Diabetes_binary"]

        # save_parquet(X_ref, y_ref, prefix="ref", output_dir=data_dir)
        logging.info(f"Data succesfully splitted to ref X,y")
        upload_to_s3(X_ref, "X_ref")
        upload_to_s3(y_ref, "y_ref")

    else:
        logging.error("Unknown target column -> data was not saved.")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, y_train = balancing_classes(X_train, y_train)
    # logging.info(f'Data succesfully splitted to test and train (8/2)')


def change_classes(df):
    df.iloc[:, 0] = df.iloc[:, 0].replace(2, 1)

    return df


if __name__ == "__main__":
    args = parse_args()
    read_dataframe(force_download=args.force_download)
