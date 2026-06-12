import pandas as pd
df = pd.read_csv('laptop_price.csv')

# Print every unique processor string in the dataset
for cpu in df['processor'].unique():
    print(cpu)
    