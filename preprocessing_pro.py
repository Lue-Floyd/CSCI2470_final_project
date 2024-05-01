import pandas as pd


def clean_and_save_csv(input_file, output_file):
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Replace 'ND'
    data.replace('ND', pd.NA, inplace=True)


    data.fillna(method='ffill', inplace=True)

    # Save
    data.to_csv(output_file, index=False)



input_file_path = 'interest_rate.csv'
output_file_path = 'cleaned_interest_rate.csv'


clean_and_save_csv(input_file_path, output_file_path)
