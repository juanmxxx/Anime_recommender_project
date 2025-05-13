# Get dataset

import pandas as pd

df = pd.read_csv('dataset/anime-dataset-2023.csv')

# Prin all columns like a list
for col in df.columns:
    print(col)

# ======= CLEANING =======


# First check the duplicated values
duplicates = df.duplicated()

# Print the number of duplicated values
print(f"Number of duplicated values: {duplicates.sum()}")

# Drop the duplicated values
df = df.drop_duplicates()


# Drop column premiered
df = df.drop(columns=['Premiered'])

# Drop the column Scored by
df = df.drop(columns=['Scored By'])

# Drop the column 'Members'
df = df.drop(columns=['Members'])

# Combine all conditions into a single filtering step
df = df[
    # Filter out rows with 'UNKNOWN' or empty values in the specified columns
    (df['Genres'] != 'UNKNOWN') & (df['Genres'] != '') &
    # Filter out rows with 'UNKNOWN' or empty values in the specified columns
    (df['Synopsis'] != 'UNKNOWN') & (df['Synopsis'] != '') &
    # Filter out rows with 'UNKNOWN' or empty values in the specified columns
    (df['Type'] != 'UNKNOWN') & (df['Type'] != '') & (df['Type'] != 'Music') &
    # Filter out rows with 'UNKNOWN' or empty values in the specified columns
    ~((df['Episodes'] == 'UNKNOWN') & (df['Status'] == 'Finished Airing'))
]


# Reorganize the 'anime_id' column avoiding the id skipping
df['anime_id'] = range(1, len(df) + 1)

# Create new CSV file with the cleaned data
df.to_csv('dataset/anime-dataset-2023-cleaned.csv', index=False)


# Print the datatype of all columns
for col in df.columns:
    print(f"{col}: {df[col].dtype}")


# Convert the first 200 rows to a dataframe
df = df.head(200)

# Print the first 5 rows of the dataframe with
print(df.head(5))
print(type(df['anime_id'].iloc[0]))  # Check the type of the first value
print(type(df['Name'].iloc[0]))  # Check the type of the first value
print(type(df['Score'].iloc[0]))  # Check the type of the first value
print(type(df['Genres'].iloc[0]))  # Check the type of the first value
print(type(df['Synopsis'].iloc[0]))  # Check the type of the first value
print(type(df['Type'].iloc[0]))  # Check the type of the first value
print(type(df['Episodes'].iloc[0]))  # Check the type of the first value
print(type(df['Aired'].iloc[0]))  # Check the type of the first value
print(type(df['Status'].iloc[0]))  # Check the type of the first value
print(type(df['English name'].iloc[0]))  # Check the type of the first value

print("Printing the type of rank, popularity, and favorites\n\n")
print(type(df['Rank'].iloc[0]))  # Check the type of the first value
# Wanna change the type of Rank to int

df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce').fillna(0).astype(int)  # Handle missing values
print(type(df['Rank'].iloc[0]))  # Check the type of the first value
print(type(df['Popularity'].iloc[0]))  # Check the type of the first value
print(type(df['Favorites'].iloc[0]))  # Check the type of the first value

print(type(df['Image URL'].iloc[0]))  # Check the type of the first value

print(all(isinstance(x, str) for x in df['Name']))

