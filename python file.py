
import time
import os
import requests
import pandas as pd

url = "https://api.openbrewerydb.org/v1/breweries"
response = requests.get(url, params={"page": 1, "per_page": 1})
data = response.json()
print(data[0])


page = 1
all_breweries = []

while True:
    response = requests.get(url, params={"page": page, "per_page": 200})
    data = response.json()
    if len(data) == 0:
        break

    for brewery in data:
        row = {}
        row["name"] = brewery["name"]
        row["brewery_type"] = brewery["brewery_type"]
        row["city"] = brewery["city"]
        row["state_province"] = brewery["state_province"]
        row["country"] = brewery["country"]
        row["longitude"] = brewery["longitude"]
        row["latitude"] = brewery["latitude"]
        all_breweries.append(row)

    print("Page", page, "done, total rows so far:", len(all_breweries))
    page = page + 1
    time.sleep(0.3)

df = pd.DataFrame(all_breweries)
print(df.shape)
print(df.head())

print("Total rows:", len(df))
print("Missing brewery_type:", df["brewery_type"].isnull().sum())
print(df["brewery_type"].value_counts())

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_data.csv", index=False)
print("Saved", len(df), "rows to data/raw_data.csv")


