import okx.MarketData as MarketData
import numpy as np
from datetime import datetime
import os
import csv


def print_hi(name):
    flag = "0"

    marketDataAPI = MarketData.MarketAPI(flag=flag)

    result1 = marketDataAPI.get_index_candlesticks(
        instId="BTC-USD",
        bar="1Dutc",
        after=1704067200000,
        before=1672531200,
        limit=100
    )
    result2 = marketDataAPI.get_index_candlesticks(
        instId="BTC-USD",
        bar="1Dutc",
        after=1695427200000,
        before=1672531200,
        limit=100
    )
    result3 = marketDataAPI.get_index_candlesticks(
        instId="BTC-USD",
        bar="1Dutc",
        after=1686787200000,
        before=1672531200,
        limit=100
    )
    result4 = marketDataAPI.get_index_candlesticks(
        instId="BTC-USD",
        bar="1Dutc",
        after=1678147200000,
        before=1672531200,
        limit=65
    )
    # get train dataset
    results = [result1, result2, result3, result4]
    train_dates = []
    for result in results:
        # data.append(result['data'])
        for row in result['data']:
            timestamp = int(row[0]) / 1000  # Convert milliseconds to seconds
            date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')  # Convert timestamp to date string
            new_row = [date] + row[1:]
            train_dates.append(new_row)

    np_data = np.array(train_dates)
    sorted_indices = np_data[:, 0].argsort()
    sorted_np_data = np_data[sorted_indices]

    output_directory = '../data'
    os.makedirs(output_directory, exist_ok=True)

    # Construct file path
    file_path = os.path.join(output_directory, 'data.csv')

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Confirm"])
        writer.writerows(sorted_np_data)

    print("Timestamp        Open    High    Low     Close   Confirm")
    for row in sorted_np_data:
        print("{:<16} {:<7} {:<7} {:<7} {:<7} {:<7}".format(*row))

    # print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

