import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with the correct delimiter
df = pd.read_csv("data/sample.csv", delimiter=';')

# Show basic stats
print(df.describe())

# Plot a histogram of the 'Age' column
df['Age'].hist(bins=20)  # Ensure you are using the correct column name ('Age', not 'age')
plt.title('Age Distribution')
plt.show()