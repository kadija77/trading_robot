import sys
import os
import pandas as pd
import numpy as np
import numpy.ma as ma
import asyncio
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from datetime import timedelta
from tqdm.asyncio import tqdm_asyncio  # For asynchronous progress bar
from asyncio import Semaphore

# Add your module path if necessary
sys.path.append('/trading_robot/src')

from exports.Hetvae.Hetvae_ohlc_prod_v4 import Hetvae_ModelWrapper
from exports.Preprocessing import Ts_preprocessing

warnings.filterwarnings("ignore")

class SignalReturnAnalyzer:
    def __init__(self, frequency='DailyOhlcPreprocessing', start_date=None, end_date=None, num_tickers=100, target_observations=10000):
        self.frequency = frequency
        self.api_keys = ['EzZFqnHBWCPK_YqTh8LB1_ZVRR3hwRvw']  # Replace with your valid Polygon.io API key
        self.api_key_index = 0
        self.modelWrapper = Hetvae_ModelWrapper()
        self.models_object = self.modelWrapper.models_object[frequency]
        self.preprocessing_args = self.modelWrapper.preprocessing_args[frequency]
        self.session = None
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_tickers = num_tickers
        self.target_observations = target_observations
        self.tickers = []
        self.all_tickers_data = {}
        self.gaps = self.get_gaps_for_frequency()
        self.results = pd.DataFrame()
        self.random_seed = 42  # For reproducibility
    
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.fetch_available_tickers()
        await self.fetch_all_tickers_data(self.start_date, self.end_date)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    def get_api_key(self):
        api_key = self.api_keys[self.api_key_index]
        return api_key

    def map_frequency_to_api_params(self):
        if self.frequency == 'MinuteOhlcPreprocessing':
            return 'minute', 15  # 15-minute bars
        elif self.frequency == 'HourOhlcPreprocessing':
            return 'hour', 1  # Hourly bars
        elif self.frequency == 'DailyOhlcPreprocessing':
            return 'day', 1  # Daily bars
        else:
            raise ValueError("Invalid frequency specified")

    def get_gaps_for_frequency(self):
        if self.frequency == 'MinuteOhlcPreprocessing':
            return [1, 2, 5, 8, 10]  # Gaps in 15-minute intervals
        elif self.frequency == 'HourOhlcPreprocessing':
            return [1, 2, 5, 8, 10]  # Gaps in hours
        elif self.frequency == 'DailyOhlcPreprocessing':
            return [1, 2, 5, 8, 10]  # Gaps in days
        else:
            raise ValueError("Invalid frequency specified")

    def get_time_deltas_for_frequency(self):
        if self.frequency == 'MinuteOhlcPreprocessing':
            # Each bar represents 15 minutes
            return [timedelta(minutes=gap * 15) for gap in self.gaps]
        elif self.frequency == 'HourOhlcPreprocessing':
            return [timedelta(hours=gap) for gap in self.gaps]
        elif self.frequency == 'DailyOhlcPreprocessing':
            return [timedelta(days=gap) for gap in self.gaps]
        else:
            raise ValueError("Invalid frequency specified")

    async def fetch_available_tickers(self):
        """
        Fetch available tickers and select random tickers.
        """
        random.seed(self.random_seed)
        print("Fetching available tickers...")
        url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={self.get_api_key()}"
        tickers = []
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                tickers.extend([item['ticker'] for item in data.get('results', [])])
            else:
                print(f"Failed to fetch tickers. Status code: {response.status}")
        # Select random tickers
        if len(tickers) >= self.num_tickers:
            self.tickers = random.sample(tickers, self.num_tickers)
        else:
            self.tickers = tickers
            print(f"Only {len(tickers)} tickers available.")
        print(f"Selected tickers: {self.tickers}")

    async def fetch_all_tickers_data(self, start_date, end_date):
        """
        Fetch full historical data for all tickers and store in a dictionary.
        """
        self.all_tickers_data = {}

        # Determine extended periods
        lookback_days = self.models_object.data_args['max_len'] * 2  # Extend lookback period
        max_future_gap = max(self.gaps)
        # Extend the end_date by a safety margin
        future_margin = max_future_gap * 2

        start_date_with_lookback = (start_date - pd.DateOffset(days=lookback_days)).strftime('%Y-%m-%d')
        end_date_with_future = (end_date + pd.DateOffset(days=future_margin)).strftime('%Y-%m-%d')
        api_frequency, multiplier = self.map_frequency_to_api_params()

        tasks = []
        for ticker in self.tickers:
            base_url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{api_frequency}/{start_date_with_lookback}/{end_date_with_future}?adjusted=true&sort=asc&limit=50000&apiKey={self.get_api_key()}'
            tasks.append(self.fetch_ticker_data(ticker, base_url))

        # Use asyncio.gather to run tasks concurrently
        results = await asyncio.gather(*tasks)

        # Store the data
        for ticker, data in results:
            if data is not None:
                self.all_tickers_data[ticker] = data

    async def fetch_ticker_data(self, ticker, url):
        async with self.session.get(url) as response:
            if response.status == 200:
                raw_data = await response.json()
                if 'results' not in raw_data or not raw_data['results']:
                    print(f"No data returned for ticker {ticker}.")
                    return ticker, None
                table = pd.DataFrame(raw_data['results'])
                table.rename(columns={
                    'o': 'Open',
                    'h': 'High',
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume',
                    't': 'Date'
                }, inplace=True)

                table['Date'] = pd.to_datetime(table['Date'], unit='ms')

                # Ensure data is sorted by date
                table.sort_values('Date', inplace=True)
                return ticker, table
            else:
                print(f"Failed to fetch data for {ticker}. Status code: {response.status}")
                return ticker, None

    async def analyze(self):
        """
        Main method to perform the analysis and prepare the tables.
        """
        # Limit the number of concurrent tasks
        semaphore = Semaphore(10)  # Adjust the concurrency limit as needed

        # Create tasks for each ticker
        tasks = [self.process_ticker_with_semaphore(ticker, semaphore) for ticker in self.tickers]

        # Use tqdm to display progress bar
        for f in tqdm_asyncio.as_completed(tasks, desc="Analyzing Tickers", total=len(tasks)):
            await f

    async def process_ticker_with_semaphore(self, ticker, semaphore):
        async with semaphore:
            await self.process_ticker(ticker)

    async def process_ticker(self, ticker):
        """
        Process a single ticker to generate the adjusted signals and future returns.
        """
        data = self.all_tickers_data.get(ticker)
        if data is None or data.empty:
            print(f"No data available for ticker {ticker}.")
            return

        # Ensure data is sorted by date
        data.sort_values('Date', inplace=True)

        # Find indices corresponding to dates within the analysis period
        analysis_indices = data[(data['Date'] >= self.start_date) & (data['Date'] <= self.end_date)].index.tolist()

        # Initialize lists to store results
        adjusted_signals = []
        dates = []
        future_returns = {f'Future_Return_{gap}': [] for gap in self.gaps}
        # Initialize lists for additional metrics
        metric_list = []

        # Get time deltas for gaps
        time_deltas = self.get_time_deltas_for_frequency()

        # For each index in the analysis period
        for idx in analysis_indices:
            # Check if target observations reached
            if len(self.results) >= self.target_observations:
                print(f"Reached target of {self.target_observations} observations.")
                return

            # Ensure we have enough historical data for the model
            start_idx = idx - self.models_object.data_args['max_len']
            if start_idx < 0:
                continue  # Skip if not enough data

            # Get data for model input
            current_data = data.iloc[start_idx:idx]
            current_date = data['Date'].iloc[idx]

            # Generate adjusted signal
            prediction_data = await self.get_prediction(ticker, current_data)
            if prediction_data is None:
                continue

            adjusted_signal = prediction_data['prediction']['aggregated_metrics']['signal']['OHLC']['adjusted_signal']

            # Extract additional metrics
            # Example: Include all aggregated metrics
            aggregated_metrics = prediction_data['prediction']['aggregated_metrics']
            # Flatten the aggregated metrics dictionary
            flat_metrics = self.flatten_dict(aggregated_metrics)
            metric_list.append(flat_metrics)

            # Compute future returns based on timestamps
            for gap, time_delta in zip(self.gaps, time_deltas):
                future_date = current_date + time_delta
                # Find the row with the future date
                future_data = data[data['Date'] >= future_date]
                if not future_data.empty:
                    future_price = future_data['Close'].iloc[0]
                    current_price = data['Close'].iloc[idx]
                    future_return = (future_price - current_price) / current_price
                    future_returns[f'Future_Return_{gap}'].append(future_return)
                else:
                    # Not enough data for this gap
                    future_returns[f'Future_Return_{gap}'].append(np.nan)

            adjusted_signals.append(adjusted_signal)
            dates.append(current_date)

        # Check if we have any results
        if not adjusted_signals:
            print(f"Not enough data for ticker {ticker}.")
            return

        # Create DataFrame for this ticker
        metrics_df = pd.DataFrame(metric_list)
        ticker_df = pd.DataFrame({
            'Ticker': ticker,
            'Date': dates,
            'Adjusted_Signal': adjusted_signals,
            **future_returns
        })
        # Combine with metrics DataFrame
        ticker_df = pd.concat([ticker_df, metrics_df.reset_index(drop=True)], axis=1)

        # Append to the main results DataFrame
        self.results = pd.concat([self.results, ticker_df], ignore_index=True)

    async def get_prediction(self, ticker, data):
        # Process the data to create input vector
        vector, ref = self.ticker_to_vector(data, ticker)

        final_output = {
            "frequency": self.frequency,
            "vectors": vector,
            "ref": [ref],
            "raw_ohlc": data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']],
        }

        # Call predict and await the result
        predictions = await self.modelWrapper.predict(final_output)

        # Include predictions in output
        final_output['prediction'] = predictions[ticker][self.frequency]['output']

        return final_output

    def ticker_to_vector(self, table, ticker):
        data_args = {}
        data_args['frequency'] = self.map_frequency_to_api_params()[0]
        data_args['vars'] = {'X': ['Open', 'High', 'Low', 'Close', 'Volume']}
        data_args['normalize'] = True
        data_args['compute_return_cols'] = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_args['genereate_technical_indicator'] = True
        data_args['box_cox_vars'] = ['Open', 'High', 'Low', 'Close', 'Volume']
        table['Ticker'] = ticker

        # Process the data
        table = Ts_preprocessing.prepare_data(table, data_args)
        table, logReturnCols = Ts_preprocessing.compute_log_returns(table, data_args['vars']['X'])
        table, TechnicalIndicators = Ts_preprocessing.Gen_technical_indicators(table, data_args)
        data_args['vars']['X'] = data_args['vars']['X'] + logReturnCols + TechnicalIndicators
        table = Ts_preprocessing.rearrange_columns(table)
        table, _ = Ts_preprocessing.apply_box_cox_transform(table, data_args['box_cox_vars'], self.preprocessing_args['lamdas_values'])
        table, _ = Ts_preprocessing.normalize_data(table, data_args['vars']['X'], scaling_model=self.preprocessing_args['scaling_model'])
        table, _ = Ts_preprocessing.handle_outliers_zscore(table, data_args['vars']['X'], outliers_dict=self.preprocessing_args['outliers_dict'])
        time = table['Date'].max()

        table['Normalized_Date'] = (table['Date'] - table['Date'].min()) / (table['Date'].max() - table['Date'].min())

        t = np.zeros(self.models_object.data_args['max_len'])
        t_trunc = table['Normalized_Date'].values[-self.models_object.data_args['max_len']:]
        t[:t_trunc.shape[0]] = t_trunc

        dim = np.empty((self.models_object.data_args['max_len'], len(self.models_object.data_args['vars']['X'])))
        dim[:] = np.nan
        dim[:table[self.models_object.data_args['vars']['X']].values.shape[0]] = table[self.models_object.data_args['vars']['X']].values[-(self.models_object.data_args['max_len']):]

        mask = np.invert(ma.masked_invalid(dim).mask)
        dim = np.nan_to_num(dim)

        x_vector = np.concatenate([dim, mask.astype(float)], axis=1)
        x_vector = np.concatenate([x_vector, np.expand_dims(t, axis=1)], axis=1)
        x_vector = np.expand_dims(x_vector, axis=0)

        return x_vector, (time, ticker)

    def flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten a nested dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

# Main execution function
async def run_analysis():

    frequencies = ['DailyOhlcPreprocessing','MinuteOhlcPreprocessing', 'HourOhlcPreprocessing']
    start_date = '2024-01-01'
    end_date = '2024-06-01'
    num_tickers = 2000  # Specify the number of tickers
    target_observations = 50000  # Specify the target number of observations

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    results = {}

    for frequency in frequencies:
        print(f"Analyzing frequency: {frequency}")
        analyzer = SignalReturnAnalyzer(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            num_tickers=num_tickers,
            target_observations=target_observations
        )

        async with analyzer:
            await analyzer.analyze()

        # Save results to CSV
        filename = f'/trading_robot/src/Analysis/ModelPerformance/signal_return_{frequency}.csv'
        analyzer.results.to_csv(filename, index=False)
        results[frequency] = analyzer.results

        print(f"Results saved to {filename}")

# Run the analysis
if __name__ == '__main__':
    asyncio.run(run_analysis())