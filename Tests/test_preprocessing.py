import pandas as pd

from Model import preprocessing_data


def test_change_types_valid():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["3", "4"]})
    converted = preprocessing_data.change_types(df)
    assert all(converted.dtypes == "int")


def test_selecting_features_binary():
    df = pd.DataFrame(
        {
            "Diabetes_binary": [0, 1],
            "BMI": [22, 30],
            "Age": [50, 60],
            "Income": [3, 4],
            "PhysHlth": [5, 10],
            "Education": [2, 3],
            "GenHlth": [4, 3],
            "MentHlth": [2, 0],
            "HighBP": [0, 1],
            "Fruits": [1, 0],
            "Extra": [1, 2],  # Should not be selected
        }
    )

    df_selected = preprocessing_data.selecting_features(df)

    expected_cols = [
        "Diabetes_binary",
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

    assert set(df_selected) == set(expected_cols)


def test_num_of_selected_features():
    num_of_columns = 10
    df = pd.DataFrame(
        {
            "Diabetes_binary": [0, 1],
            "BMI": [22, 30],
            "Age": [50, 60],
            "Income": [3, 4],
            "PhysHlth": [5, 10],
            "Education": [2, 3],
            "GenHlth": [4, 3],
            "MentHlth": [2, 0],
            "HighBP": [0, 1],
            "Fruits": [1, 0],
            "Extra": [1, 2],  # Should not be selected
        }
    )

    df_selected = preprocessing_data.selecting_features(df)

    assert num_of_columns == len(df_selected.columns)


def test_change_classes():
    test_df = pd.DataFrame({"Diabetes": [0, 1, 1, 2, 2]})
    expected_df = pd.DataFrame({"Diabetes": [0, 1, 1, 1, 1]})

    test_df = preprocessing_data.change_classes(test_df)

    assert (test_df == expected_df).all().all()
