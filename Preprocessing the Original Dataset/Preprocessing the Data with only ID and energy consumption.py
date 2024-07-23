'''
This Python code is used to preprocess the original SmartMeter Energy 
Consumption Data in London Households. The dataset was first downloaded
from the website of London Datastore available at 
https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
and saved locally as a csv file.

Here we will drop the columns for hour, day of the year, and is_weekend to accelerate computation.
'''
from sklearn.preprocessing import LabelEncoder
import pandas as pd

full_data = pd.read_csv("CC_LCL-FullData.csv")

# Convert datetime column to pandas DateTime type
full_data['DateTime'] = pd.to_datetime(full_data['DateTime'])

# Extract day of the year
full_data['dayoftheyear'] = full_data['DateTime'].dt.dayofyear

# Filter the data to year 2013
full_data['year'] = full_data['DateTime'].dt.year
full_data = full_data[full_data['year'] == 2013]
full_data.drop(columns=["year"], inplace=True)

# Extract hour
full_data['hour'] = full_data['DateTime'].dt.hour

# Drop DateTime column as it is not needed anymore
full_data.drop(columns=["DateTime"], inplace=True)

# Select only standard customers 
full_data = full_data[full_data["stdorToU"] == "Std"]
full_data.drop(columns=["stdorToU"], inplace=True)
# After filtering the dataset the stdorToU column is redundant, for this reason we remove it
# inplace = True is used to modify existing dataframe instead of creating a new one.

# Encode the ID of meters to a numerical value
label_encoder_lclid = LabelEncoder()
full_data['LCLid'] = label_encoder_lclid.fit_transform(full_data['LCLid'])
full_data['LCLid'] = full_data['LCLid'].astype('int16')

# Ensure that target does not contain NULLs and is numerric 
full_data['KWH/hh (per half hour) '] = pd.to_numeric(full_data['KWH/hh (per half hour) '], errors='coerce').fillna(0.0).astype('float16')

# Aggregate the dataset into hourly energy consumption instead of half hourly
hourly_data = full_data.groupby(['LCLid','dayoftheyear','hour'])['KWH/hh (per half hour) '].sum().reset_index()
hourly_data = hourly_data.rename(columns={'KWH/hh (per half hour) ': 'KWH/hh (per hour) '})

# Drop the columns for hour and day of the year
hourly_data.drop(columns=["hour"], inplace=True)
hourly_data.drop(columns=["dayoftheyear"], inplace=True)

# Save the DataFrame to a CSV file
hourly_data.to_csv('Preprocessed_data2013_2.csv', index=False)

