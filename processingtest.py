import pandas as pd

def augment_data_with_weekend_rates(input_file, output_file, start_date):

    data = pd.read_csv(input_file, header=None)
    rates = data.iloc[:, 0]


    dates = pd.date_range(start=start_date, periods=len(rates), freq='B')


    augmented_data = []


    day_counter = 0
    while day_counter < len(rates):
        current_date = dates[day_counter]
        current_rate = rates.iloc[day_counter]
        augmented_data.append({'date': current_date, 'rate': current_rate})

        if current_date.dayofweek == 4:  # Check if it's Friday
            # Append Saturday and Sunday with the same rate as Friday
            augmented_data.append({'date': current_date + pd.Timedelta(days=1), 'rate': current_rate})
            augmented_data.append({'date': current_date + pd.Timedelta(days=2), 'rate': current_rate})
            day_counter += 1  # Move to the next rate in the list
        day_counter += 1  # Normal increment for other weekdays


    augmented_data_df = pd.DataFrame(augmented_data)


    augmented_data_df.to_csv(output_file, index=False)


input_file_path = 'cleaned_interest_rate.csv'
output_file_path = 'final_rate.csv'
start_date = '2018-1-10'  # Specify the start date


augment_data_with_weekend_rates(input_file_path, output_file_path, start_date)
