#Task 1 - Data Acquisition (No API Key Required)
import time
import os
import requests
import pandas as pd

#Step 1: Choose your API
url = "https://api.openbrewerydb.org/v1/breweries"
# Step 2: Explore before you code
#Step 3: Write the fetch script
response = requests.get(url, params={"page": 1, "per_page": 1})
data = response.json()
print(data[0])


page = 1
all_breweries = []

while True:  #keep getting pages until empty
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


#Task 2 - EDA and Data Cleaning
#Step 1: First look
df = pd.read_csv("data/raw_data.csv")

print(df.info())
print(df.describe())

missing = df.isna().sum()
print(missing)

duplicates = df.duplicated().sum()
print(duplicates)


dupes = df[df.duplicated()]
print(dupes)

extreme_lat = df[df["latitude"] < -80]
print(extreme_lat)

check = df[df["name"] == "Boston Beer Co"]
print(check)

check2 = df[df["name"] == "Goat Island Brewing"]
print(check2)

pd.set_option("display.max_columns", None)
print(df[df["name"] == "Goat Island Brewing"])

#Step 2: Visualize before you touch anything
import matplotlib.pyplot as plt
counts = df["brewery_type"].value_counts()
print(counts)
counts.plot(kind="bar")
plt.title("Number of breweries per type")
plt.xlabel("Brewery type")
plt.ylabel("Count")
plt.show()


import seaborn as sns
numeric_cols = df[["longitude", "latitude"]]
corr = numeric_cols.corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.title("Correlation between numeric features")
plt.show()


df.boxplot(column="latitude", by="brewery_type")
plt.title("Latitude by brewery type")
plt.show()


sns.scatterplot(data=df, x="longitude", y="latitude", hue="brewery_type")
plt.title("Brewery locations by type")
plt.show()


df.boxplot(column="longitude", by="brewery_type")
plt.title("Longitude by brewery type")
plt.show()

#Step 3: Clean, with justification for each decision

#Fix the swapped longitude/latitude for Goat Island Brewing
#justification: coordinates were swapped by mistake, not real data
row_to_fix = df["name"] == "Goat Island Brewing"
old_longitude = df.loc[row_to_fix, "longitude"].values[0]
old_latitude = df.loc[row_to_fix, "latitude"].values[0]
df.loc[row_to_fix, "longitude"] = old_latitude
df.loc[row_to_fix, "latitude"] = old_longitude
print(df[df["name"] == "Goat Island Brewing"])

#Drop rows with missing longitude or latitude
#justification: making up fake ones would be misleading, 
#and only 20% of rows are affected
print("Rows before:", len(df))
df = df.dropna(subset=["longitude", "latitude"])
print("Rows after:", len(df))

#Drop exact duplicate rows
#justification: exact duplicate rows
print("Rows before:", len(df))
df = df.drop_duplicates()
print("Rows after:", len(df))

#Merge rare classes into "other"
#justification: classes with very few examples can break splitting later
counts = df["brewery_type"].value_counts()
print(counts)
rare_types = counts[counts < 50].index
print("Rare types being merged:", list(rare_types))
df["brewery_type"] = df["brewery_type"].replace(rare_types, "other")
print(df["brewery_type"].value_counts())

#encode the country column
#justification: models need numbers, not text, so need to become 0-1 columns 
df = pd.get_dummies(df, columns=["country"])
print(df.head())

#Step 4: Save your output
df.to_csv("data/clean_data.csv", index=False)
print("Saved cleaned data with", len(df), "rows")


#Task 3 - Classification Model
#Step 1: Prepare features and target
from sklearn.preprocessing import LabelEncoder

TARGET = "brewery_type"
country_cols = [c for c in df.columns if c.startswith("country_")]
FEATURES = ["longitude", "latitude"] + country_cols

X = df[FEATURES]
y = df[TARGET]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Step 2: Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

#Step 3: Scale features (needed for linear/distance-based models)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Step 4: Train at least two models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
}

#Step 5: Evaluate and compare
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    print(name, acc, prec, rec, f1)
    result_row = {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    results.append(result_row)

from sklearn.metrics import confusion_matrix
#random forest had the best macro F1 score
#0.21 for logistic regression, 0.24 for random forest
best_name = "random_forest"
best_model = models[best_name]
best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds, labels=range(len(le.classes_)))
print(cm)

#Step 6: Save everything the dashboard will need
import joblib
import json

importances = {}
for i in range(len(FEATURES)):
    feature_name = FEATURES[i]
    importance_value = best_model.feature_importances_[i]
    importances[feature_name] = importance_value

joblib.dump(best_model, "model.pkl") #save model for the dashboard
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

metrics = {}
metrics["all_model_results"] = results
metrics["best_model"] = best_name
metrics["confusion_matrix"] = cm.tolist()
metrics["class_labels"] = le.classes_.tolist()
metrics["feature_importance"] = importances
metrics["features"] = FEATURES

f = open("model_metrics.json", "w")
json.dump(metrics, f, indent=2)
f.close()

print("Saved model, scaler, encoder, and metrics")

