import os
import sys
import joblib

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
col_sentiments = db["sentiments"]

sentiments = [s["label"] for s in col_sentiments.find({})]

X, y = [], []
docs = list(col_labeled.find({}))

for d in docs:
    text = (d.get("title", "") + " " + d.get("text", "")).strip()
    topics = d.get("topics", [])
    sent = d.get("sentiments", [])
    for t, s in zip(topics, sent):
        X.append(f"{text} [TOPIC:{t}]")
        y.append(s)

# Обучение, тестирование
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=200))
])

clf.fit(X_train, y_train)

# Оценка
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=sentiments))

# Сохранение
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/sentiment_model.pkl")

print("Модель по тональностям сохранена в models/sentiment_model.pkl")