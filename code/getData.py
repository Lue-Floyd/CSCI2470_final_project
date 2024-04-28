import okx.MarketData as MarketData
import numpy as np
from datetime import datetime
import os
import csv

def get_all_data(inst_id, bar, after_time, before_time, limit=100):
    flag = "0"

    marketDataAPI = MarketData.MarketAPI(flag=flag)
    results = []
    while after_time >= before_time:
        result = marketDataAPI.get_history_candlesticks(
            instId=inst_id,
            bar=bar,
            after=after_time,
            before=before_time,
            limit=limit
        )
        results.append(result)
        after_time -= 8640000000
    return results


def print_hi(name):
    results = get_all_data("BTC-USDT", "1Dutc", 1704067200000, 1514764800000)
    train_dates = []
    for result in results:
        # data.append(result['data'])
        for row in result['data']:
            timestamp = int(row[0]) / 1000  # Convert milliseconds to seconds
            date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')  # Convert timestamp to date string
            new_row = [date] + row[1:-2]
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
        writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Vol", "VolCcy"])
        writer.writerows(sorted_np_data)

    print("Timestamp        Open    High    Low     Close   vol            volCcy")
    for row in sorted_np_data:
        print("{:<16} {:<7} {:<7} {:<7} {:<7} {:<14} {:<17}".format(*row))

    # print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

