# Experiment 1: FIND-S Algorithm

import pandas as pd

# Training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Warm', 'Warm'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Training Data:")
print(df)

# FIND-S algorithm
hypothesis = ['Ø'] * 6

for i in range(len(df)):
    if df.iloc[i]['EnjoySport'] == 'Yes':
        for j in range(6):
            if hypothesis[j] == 'Ø':
                hypothesis[j] = df.iloc[i, j]
            elif hypothesis[j] != df.iloc[i, j]:
                hypothesis[j] = '?'

print("\nMost Specific Hypothesis:")
print(hypothesis)
