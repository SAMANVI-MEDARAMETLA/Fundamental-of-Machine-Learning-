# Experiment 2: Candidate-Elimination Algorithm

import pandas as pd

# Create training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Warm', 'Warm'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('training_data.csv', index=False)

# Read CSV file
df = pd.read_csv('training_data.csv')

print("Training Data:")
print(df)

# Attributes
attributes = list(df.columns[:-1])
n = len(attributes)

# Initial Specific and General hypotheses
S = ['Ø'] * n
G = [['?'] * n]

# Candidate-Elimination Algorithm
for i in range(len(df)):

    example = list(df.iloc[i][:-1])
    target = df.iloc[i][-1]

    # Positive example
    if target == 'Yes':

        for j in range(n):

            if S[j] == 'Ø':
                S[j] = example[j]

            elif S[j] != example[j]:
                S[j] = '?'

        # Remove inconsistent hypotheses from G
        G = [
            g for g in G
            if all(g[j] == '?' or g[j] == S[j] for j in range(n))
        ]

    # Negative example
    else:

        new_G = []

        for g in G:

            for j in range(n):

                if g[j] == '?' and S[j] != '?' and S[j] != 'Ø':

                    new_g = g.copy()
                    new_g[j] = S[j]

                    if new_g not in new_G:
                        new_G.append(new_g)

        G = new_G

# Display results
print("\nSpecific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")
for g in G:
    print(g)
