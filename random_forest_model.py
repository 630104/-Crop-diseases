import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the environmental data
env_data = pd.read_csv('path/to/environmental/data.csv')

# Preprocess the data
env_data.dropna(inplace=True)
env_data['date'] = pd.to_datetime(env_data['date'])
env_data.set_index('date', inplace=True)

# Resample the data to daily frequency
env_data_daily = env_data.resample('D').mean()

# Merge the environmental data with the image data
merged_data = pd.merge(image_data, env_data_daily, on='date')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(merged_data.drop('disease', axis=1), merged_data['disease'], test_size=0.2, random_state=42)

# Define the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.3f}')