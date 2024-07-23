'''
This Python code is used to preprocess the original SmartMeter Energy 
Consumption Data in London Households. The dataset was first downloaded
from the website of London Datastore available at 
https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
and saved locally as a csv file.
'''


from sklearn.preprocessing import LabelEncoder
import pandas as pd

full_data = pd.read_csv("CC_LCL-FullData.csv")

# The datetime column is converted to pandas datetime type to extract day of year, hour
# and restrict the data to year 2013.
full_data['DateTime'] = pd.to_datetime(full_data['DateTime'])

full_data['year'] = full_data['DateTime'].dt.year
full_data = full_data[full_data['year'] == 2013]
full_data.drop(columns=["year"], inplace=True)

# The day of year is from 1st of January 2013 to 31st of December 2013
full_data['dayoftheyear'] = full_data['DateTime'].dt.dayofyear

# Create a new binary column indicating weekends.
full_data['is_weekend'] = full_data['DateTime'].dt.dayofweek >= 5

full_data['hour'] = full_data['DateTime'].dt.hour

# Drop DateTime column as it is not needed anymore
full_data.drop(columns=["DateTime"], inplace=True)

# Restrict the dataset to standard customers
full_data = full_data[full_data["stdorToU"] == "Std"]
full_data.drop(columns=["stdorToU"], inplace=True)

# Encode the ID of meters as numerical values
label_encoder_lclid = LabelEncoder()
full_data['LCLid'] = label_encoder_lclid.fit_transform(full_data['LCLid'])
full_data['LCLid'] = full_data['LCLid'].astype('int16')


# Ensure that target does not contain NULLs and is numeric
full_data['KWH/hh (per half hour) '] = pd.to_numeric(full_data['KWH/hh (per half hour) '], errors='coerce').fillna(0.0).astype('float16')

# Aggregate the half hourly energy consumption data to hourly.
hourly_data = full_data.groupby(['LCLid', 'dayoftheyear', 'hour', 'is_weekend'])['KWH/hh (per half hour) '].sum().reset_index()
hourly_data = hourly_data.rename(columns={'KWH/hh (per half hour) ': 'KWH/hh (per hour) '})

# Save the DataFrame to a CSV file
hourly_data.to_csv('Preprocessed_data2013.csv', index=False)
