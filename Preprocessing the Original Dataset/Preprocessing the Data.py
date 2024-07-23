from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Read original dataset
full_data = pd.read_csv("CC_LCL-FullData.csv")

# Convert datetime column to pandas DateTime type
full_data['DateTime'] = pd.to_datetime(full_data['DateTime'])

# Extract day of the year
full_data['dayoftheyear'] = full_data['DateTime'].dt.dayofyear

# Extract year 2013
full_data['year'] = full_data['DateTime'].dt.year
full_data = full_data[full_data['year'] == 2013]
full_data.drop(columns=["year"], inplace=True)


# Check if the day is a weekend (Saturday=5, Sunday=6)
full_data['is_weekend'] = full_data['DateTime'].dt.dayofweek >= 5

# Extract hour
full_data['hour'] = full_data['DateTime'].dt.hour

# Drop DateTime column as it is not needed anymore
full_data.drop(columns=["DateTime"], inplace=True)

# Select only standard customers
full_data = full_data[full_data["stdorToU"] == "Std"]
full_data.drop(columns=["stdorToU"], inplace=True)

# Encode the ID of meters
label_encoder_lclid = LabelEncoder()
full_data['LCLid'] = label_encoder_lclid.fit_transform(full_data['LCLid'])
full_data['LCLid'] = full_data['LCLid'].astype('int16')


# Ensure that target does not contain NULLs and is numeric
full_data['KWH/hh (per half hour) '] = pd.to_numeric(full_data['KWH/hh (per half hour) '], errors='coerce').fillna(0.0).astype('float16')

# Calculate hourly consumption
hourly_data = full_data.groupby(['LCLid', 'dayoftheyear', 'hour', 'is_weekend'])['KWH/hh (per half hour) '].sum().reset_index()
hourly_data = hourly_data.rename(columns={'KWH/hh (per half hour) ': 'KWH/hh (per hour) '})

# Save the DataFrame to a CSV file
hourly_data.to_csv('Preprocessed_data2013.csv', index=False)
