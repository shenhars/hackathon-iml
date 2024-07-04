import pandas as pd

file = pd.read_csv('./train_bus_schedule.csv', encoding='iso8859-8')
print(file['cluster'])