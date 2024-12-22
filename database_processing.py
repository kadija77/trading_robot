from utils.decorators.logger import Logger
from utils.decorators.background import Thread
from utils.classes.singleton import SingletonMeta
from utils.entity.stock_symbol import StocksSymbolEntity
from utils.entity.ohlc_daily_table import OhlcDailyEntity
from utils.entity.ohlc_hour_table import OhlcHourlyEntity
from utils.entity.ohlc_minute_table import OhlcMinuteEntity
from utils.entity.daily_preprocess_table import DailyPreprocessEntity
from utils.entity.hour_preprocess_table import HourPreprocessEntity
from utils.entity.minute_preprocess_table import MinutePreprocessEntity
from data.Preprocessing.Preprocessing import preporcess_data
import numpy as np
import multiprocessing
import logging
from tqdm import tqdm
import datetime


@Logger(level=logging.DEBUG)
class databasePreprocessing(metaclass=SingletonMeta):

    # FULL DATABASE PREPROCESSING FUNCTIONS
    def full_database_preprocess(self):
        stocks_symbol_list = StocksSymbolEntity().select(keys=['ticker'])
        chunked_stocks = np.array_split(
            stocks_symbol_list, multiprocessing.cpu_count())
        threads = []
        for chunk in chunked_stocks:
            threads.append(self.start_full_preprocessing(chunk))

        for thread in threads:
            thread.join()

    async def start_full_preprocessing(self, chunk: list):
        return False

    # APPENDED DATABASE PREPROCESSING FUNCTIONS
    def additional_database_preprocess(self):
        self.logger.info("Starting Additional Database Pre Processing")
        stocks_symbol_list = list(
            map(lambda s: s['ticker'], StocksSymbolEntity().select(
                keys=['ticker']))
        )
        self.logger.info("Total Number of Stocks => %s",
                         len(stocks_symbol_list))
        chunked_stocks = np.array_split(
            stocks_symbol_list, multiprocessing.cpu_count())
        threads = []
        for chunk in chunked_stocks:
            threads.append(self.start_additional_preprocessing(chunk))

        for thread in threads:
            thread.join()

    @Thread
    async def start_additional_preprocessing(self, chunk: list):
        for stock in tqdm(chunk):
            self.preprocess_additional_daily(stock)
            self.preprocess_additional_hourly(stock)
            self.preprecess_additional_minute(stock)

    def preprocess_additional_daily(self, stock):
        try:
            last_inserted_preprocess = DailyPreprocessEntity().select(
                keys=['Date'], findBy=f"WHERE Ticker='{stock}' ORDER BY Date DESC LIMIT 1")
            if len(last_inserted_preprocess) == 0:
                self.logger.info(
                    "DAILY PREPROCESSING STOCK UP TO DATE => %s", stock)
                return
            converted_date = datetime.datetime.strptime(
                last_inserted_preprocess[0]['Date'], '%Y-%m-%d %H:%M:%S')
            all_records_without_preprocess = OhlcDailyEntity().select(
                keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date > '{converted_date}' ORDER BY Date ASC")
            self.logger.info(
                "STARTING DAILY PREPROCESS FOR {%s} => NUMBER OF ROWS MISSING %s", stock, len(all_records_without_preprocess))
            for record in all_records_without_preprocess:
                needed_records = OhlcDailyEntity().select(
                    keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date < '{record['Date']}' ORDER BY Date DESC LIMIT 25")
                needed_records.reverse()
                needed_records.append(record)
                preporcessed_data = preporcess_data(
                    needed_records, Preprocessed_Name='DailyOhlcPreprocessing')
                DailyPreprocessEntity().insert(preporcessed_data)
            self.logger.info("COMPLETED DAILY PREPROCESS FOR => %s", stock)
        except:
            self.logger.error(
                "Daily PREPROCESS ERROR FOR => %s", stock, exc_info=True)

    def preprocess_additional_hourly(self, stock):
        try:
            last_inserted_preprocess = HourPreprocessEntity().select(
                keys=['Date'], findBy=f"WHERE Ticker='{stock}' ORDER BY Date DESC LIMIT 1")
            if len(last_inserted_preprocess) == 0:
                self.logger.info(
                    "HOURLY PREPROCESSING STOCK UP TO DATE => %s", stock)
                return
            converted_date = datetime.datetime.strptime(
                last_inserted_preprocess[0]['Date'], '%Y-%m-%d %H:%M:%S')
            all_records_without_preprocess = OhlcHourlyEntity().select(
                keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date > '{converted_date}' ORDER BY Date ASC")
            self.logger.info(
                "STARTING HOURLY PREPROCESS FOR {%s} => NUMBER OF ROWS MISSING %s", stock, len(all_records_without_preprocess))
            for record in all_records_without_preprocess:
                needed_records = OhlcHourlyEntity().select(
                    keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date < '{record['Date']}' ORDER BY Date DESC LIMIT 25")
                needed_records.reverse()
                needed_records.append(record)
                preporcessed_data = preporcess_data(
                    needed_records, Preprocessed_Name='HourOhlcPreprocessing')
                HourPreprocessEntity().insert(preporcessed_data)
            self.logger.info("COMPLETED HOURLY PREPROCESS FOR => %s", stock)
        except:
            self.logger.error(
                "Hourly PREPROCESS ERROR FOR => %s", stock, exc_info=True)

    def preprecess_additional_minute(self, stock):
        try:
            last_inserted_preprocess = MinutePreprocessEntity().select(
                keys=['Date'], findBy=f"WHERE Ticker='{stock}' ORDER BY Date DESC LIMIT 1")
            if len(last_inserted_preprocess) == 0:
                self.logger.info(
                    "HOURLY PREPROCESSING STOCK UP TO DATE => %s", stock)
                return
            converted_date = datetime.datetime.strptime(
                last_inserted_preprocess[0]['Date'], '%Y-%m-%d %H:%M:%S')
            all_records_without_preprocess = OhlcMinuteEntity().select(
                keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date > '{converted_date}' ORDER BY Date ASC")
            self.logger.info(
                "STARTING MINUTE PREPROCESS FOR {%s} => NUMBER OF ROWS MISSING %s", stock, len(all_records_without_preprocess))
            for record in all_records_without_preprocess:
                needed_records = OhlcMinuteEntity().select(
                    keys=['*'], findBy=f"WHERE Ticker='{stock}' AND Date < '{record['Date']}' ORDER BY Date DESC LIMIT 25")
                needed_records.reverse()
                needed_records.append(record)
                preporcessed_data = preporcess_data(
                    needed_records, Preprocessed_Name='MinuteOhlcPreprocessing')
                MinutePreprocessEntity().insert(preporcessed_data)
            self.logger.info("COMPLETED MINUTE PREPROCESS FOR => %s", stock)
        except:
            self.logger.error(
                "MINUTE PREPROCESS ERROR FOR => %s", stock, exc_info=True)



