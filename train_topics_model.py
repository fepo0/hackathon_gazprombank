import os
import sys
import joblib

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# БД
user = input("Введите свой username: ").strip()
path = "autorization.txt"
uri = None

if path in os.listdir(os.getcwd()):
    with open(file=path, mode="r", encoding="utf-8") as file:
        text = file.read().strip()
        for line in text.splitlines():
            parts = line.strip().split(":", 1)
            if len(parts) != 2:
                continue
            if parts[0] == user:
                password = parts[1]
                uri = (f"mongodb+srv://{user}:{password}@myclaster.qwyjq9x.mongodb.net/?"
                       f"retryWrites=true&w=majority&appName=myclaster")
                break
        else:
            print("Нет такого username в базе")

if uri is None:
    print("Нет uri")
    sys.exit(1)

client = MongoClient(uri, server_api=ServerApi("1"))
db = client["banki_db"]

col_labeled = db["reviews_labeled"]
col_clusters = db["clusters"]

docs = list(col_labeled.find({}))
texts = []
labels = []

for d in docs:
    text = (d.get("title", "") + " " + d.get("text", "")).strip()
    topics = d.get("topics", [])
    if text and topics:
        texts.append(text)
        labels.append(topics)

# multi-label
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

# TF_IDF
X_train, X_test, y_train, y_test = train_test_split(texts, Y, test_size=0.2, random_state=42)

clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200))),
])

clf.fit(X_train, y_train)

# Оценка
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Сохранение модели
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/topics_model.pkl")
joblib.dump(mlb, "models/topics_mlb.pkl")

print("Модель по услугам(товарам) сохранена в models/topics_model.pkl")