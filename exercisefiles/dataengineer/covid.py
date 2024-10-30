import pandas as pd

# CSV-Datei einlesen
df = pd.read_csv('tested_worldwide.csv')

# Erste Spalte ausgeben
first_column = df.iloc[:, 0]
#print(first_column)