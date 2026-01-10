import pandas as pd

def display_unique_counts(df, column):
    print(f"Unique values and counts for '{column}':\n")
    for val, count in df[column].value_counts().items():
        print(f"- {val}: {count}")


#def calculate_dir(df, column, label,):
    