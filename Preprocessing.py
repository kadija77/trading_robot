import os
from exports.Preprocessing.Ts_preprocessing import prepare_data, compute_log_returns, Gen_technical_indicators, rearrange_columns, apply_box_cox_transform, normalize_data, handle_outliers_zscore
from utils.decorators.logger import Logger
import pandas as pd
from tqdm import tqdm
import pickle
import logging


@Logger(logging.INFO, file_name="preprocess")
def preporcess_data(data: list = [], Preprocessed_Name=None) -> pd.DataFrame:
    if Preprocessed_Name is None:
        return
    daily_ohlc_data_args = {}

    daily_ohlc_data_args['Preprocessing_Name'] = 'DailyOhlcPreprocessing'
    daily_ohlc_data_args['frequency'] = 'day'
    daily_ohlc_data_args['rows_per_ticker'] = 500
    daily_ohlc_data_args['DataPctSample'] = 0.8
    daily_ohlc_data_args['vars'] = {
        'X': ['Open', 'High', 'Low', 'Close', 'Volume']}
    daily_ohlc_data_args['normalize'] = True
    daily_ohlc_data_args['compute_return_cols'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    daily_ohlc_data_args['genereate_technical_indicator'] = True
    daily_ohlc_data_args['box_cox_vars'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    daily_ohlc_data_args['multitask_args'] = {
        'IN':  {'Y_clas': [
            ['Close', 10, 5, 3],
            ['Close', 12, 3, 3],
            ['Close', 14, 2, 3],
            ['Close', 16, 1, 3],
            ['High', 16, 1, 3],
            ['Low', 16, 1, 3],
            ['Open', 16, 1, 3],
            ['Volume', 16, 1, 3],
            ['Volume', 14, 2, 3],
            ['Volume', 12, 3, 3],]}
    }

    hour_ohlc_data_args = {}
    hour_ohlc_data_args['Preprocessing_Name'] = 'HourOhlcPreprocessing'
    hour_ohlc_data_args['frequency'] = 'hour'
    hour_ohlc_data_args['rows_per_ticker'] = 500
    hour_ohlc_data_args['DataPctSample'] = 0.8
    hour_ohlc_data_args['vars'] = {
        'X': ['Open', 'High', 'Low', 'Close', 'Volume']}
    hour_ohlc_data_args['normalize'] = True
    hour_ohlc_data_args['compute_return_cols'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    hour_ohlc_data_args['genereate_technical_indicator'] = True
    hour_ohlc_data_args['box_cox_vars'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    hour_ohlc_data_args['multitask_args'] = {
        'IN':  {'Y_clas': [
            ['Close', 10, 5, 3],
            ['Close', 15, 3, 3],
            ['Close', 20, 2, 3],
            ['Close', 30, 1, 3],
            ['High', 30, 1, 3],
            ['Low', 30, 1, 3],
            ['Open', 30, 1, 3],
            ['Volume', 30, 1, 3],
            ['Volume', 20, 2, 3],
            ['Volume', 15, 3, 3],]}
    }

    minute_ohlc_data_args = {}
    minute_ohlc_data_args['Preprocessing_Name'] = 'MinuteOhlcPreprocessing'
    minute_ohlc_data_args['frequency'] = 'minute'
    minute_ohlc_data_args['rows_per_ticker'] = 800
    minute_ohlc_data_args['DataPctSample'] = 0.6
    minute_ohlc_data_args['vars'] = {
        'X': ['Open', 'High', 'Low', 'Close', 'Volume']}
    minute_ohlc_data_args['normalize'] = True
    minute_ohlc_data_args['compute_return_cols'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    minute_ohlc_data_args['genereate_technical_indicator'] = True
    minute_ohlc_data_args['box_cox_vars'] = [
        'Open', 'High', 'Low', 'Close', 'Volume']
    minute_ohlc_data_args['multitask_args'] = {
        'IN':  {'Y_clas': [
            ['Close', 10, 60, 3],
            ['Close', 15, 30, 3],
            ['Close', 20, 15, 3],
            ['Close', 30, 1, 3],
            ['High', 30, 1, 3],
            ['Low', 30, 1, 3],
            ['Open', 30, 1, 3],
            ['Volume', 30, 1, 3],
            ['Volume', 20, 2, 3],
            ['Volume', 15, 3, 3],]}
    }
    if Preprocessed_Name == "DailyOhlcPreprocessing":
        data_args = daily_ohlc_data_args
    if Preprocessed_Name == "HourOhlcPreprocessing":
        data_args = hour_ohlc_data_args
    if Preprocessed_Name == "MinuteOhlcPreprocessing":
        data_args = minute_ohlc_data_args

    try:
        pickle_obj = load_preprocessing_dict(Preprocessed_Name)

        frame_data = pd.DataFrame(data)

        OHLC_data = prepare_data(frame_data, data_args)

        OHLC_data, logReturnCols = compute_log_returns(
            OHLC_data, data_args['vars']['X'])

        OHLC_data, TechnicalIndicators = Gen_technical_indicators(
            OHLC_data, data_args)

        data_args['vars']['X'] = data_args['vars']['X'] + \
            logReturnCols + TechnicalIndicators

        OHLC_data = rearrange_columns(OHLC_data)

        OHLC_data, _ = apply_box_cox_transform(
            OHLC_data, data_args['box_cox_vars'], lambda_values=pickle_obj['lamdas_values'])

        OHLC_data, _ = normalize_data(
            OHLC_data, data_args['vars']['X'], scaling_model=pickle_obj['scaling_model'])

        OHLC_data, _ = handle_outliers_zscore(
            OHLC_data, data_args['vars']['X'], outliers_dict=pickle_obj['outliers_dict'])

        return OHLC_data
    except:
        preporcess_data.logger.error("", exc_info=True)


# @Inject(singlestore_connection)
# @Logger(logging.INFO)
# def load_dataframe_to_singlestore(dataframe, table_name):
#     load_dataframe_to_singlestore.logger.info(
#         "Loading dataframe to SingleStore")
#     connection = load_dataframe_to_singlestore.deps['singlestore_connection'].getConnection(
#     )
#     cursor: Cursor = connection.cursor()
#     if cursor.execute(f"SHOW TABLES LIKE '{table_name}'") == 0:
#         cursor.execute(
#             f"CREATE TABLE {table_name} ({', '.join([f'{col} {get_column_type(dataframe[col])}' for col in dataframe.columns])})")

#     # Prepare the INSERT query
#     column_names = ', '.join(dataframe.columns)
#     placeholders = ', '.join(['%s' for _ in dataframe.columns])
#     insert_query = f"INSERT IGNORE INTO {table_name} ({column_names}) VALUES ({placeholders})"
#     total_rows = len(dataframe)
#     for _, row in tqdm(dataframe.iterrows(), total=total_rows, unit='%'):
#         cleaned_row = row.where(pd.notnull(row), None)
#         cursor.execute(insert_query, tuple(cleaned_row))
#     connection.commit()
#     load_dataframe_to_singlestore.logger.info(
#         f"Loaded {total_rows} rows to {table_name}")


@Logger(logging.INFO, file_name="preprocess")
def save_pickle(object, path):
    # Open the file in binary write mode
    with open(path, 'wb') as file:
        # Dump the object to the file
        pickle.dump(object, file)
    save_pickle.logger.info(f"Object saved successfully to {path}")


def get_column_type(column):
    if column.dtype == 'int64':
        return 'BIGINT'
    elif column.dtype == 'float64':
        return 'DOUBLE'
    elif column.dtype == 'datetime64':
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'


def check_file_exists_raise_if_does(filenName):
    """
    Check if the specified file exists. Raise an exception if it does.

    Parameters:
    - file_path (str): The path to the file to check.

    Raises:
    - Exception: If the file exists.
    """

    file_path = f'/trading_robot/models/Preprocessing_Artifacts/{filenName}.pkl'
    if os.path.exists(file_path) and os.path.isfile(file_path):
        raise Exception(f"File {file_path} already exists.")


def load_preprocessing_dict(Preprocessed_Name):
    path = f'/trading_robot/processing/{Preprocessed_Name}.pkl'
    # Open the file in binary read mode
    with open(path, 'rb') as file:
        # Load the object from the file
        loaded_object = pickle.load(file)

    return loaded_object
