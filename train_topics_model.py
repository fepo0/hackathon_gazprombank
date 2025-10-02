import os
import joblib
from typing import Dict, List

from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import MONGO_URI, DB_NAME


def get_db():
    uri = MONGO_URI or "mongodb://localhost:27017/"
    dbname = DB_NAME or "banki_db"
    client = MongoClient(uri)
    return client[dbname]


def canonical_topic_mapping() -> Dict[str, str]:
    mapping = {
        "бизнес мобильное приложение": "Мобильное приложение",
        "мобильное приложение": "Мобильное приложение",
        "дистанционное обслуживание физлиц": "Обслуживание",
        "обслуживание физических лиц": "Обслуживание",
        "расчетно-кассовое обслуживание": "Обслуживание",
        "обслуживание": "Обслуживание",
        "дебетовая карта": "Дебетовые карты",
        "дебетовые карты": "Дебетовые карты",
        "банковская карта": "Дебетовые карты",
        "кредитная карта": "Кредитные карты",
        "кредитные карты": "Кредитные карты",
    }
    return {k.lower(): v for k, v in mapping.items()}


def expand_synonyms_seed() -> Dict[str, List[str]]:
    return {
        "Мобильное приложение": [
            "приложение", "мобильный банк", "интерфейс", "функции", "удобство",
            "зависает", "тормозит", "обновление", "версия", "баг", "ошибка",
            "бизнес мобильное приложение", "приложение банка"
        ],
        "Обслуживание": [
            "обслуживание", "персонал", "менеджер", "консультант", "отделение",
            "очередь", "вежливость", "сервис", "поддержка",
            "обслуживание физических лиц", "расчетно-кассовое обслуживание",
        ],
        "Дебетовые карты": [
            "дебетовая карта", "дебетовые карты", "карточка", "банковская карта",
            "пластик", "pin-код", "банкомат"
        ],
        "Кредитные карты": [
            "кредитная карта", "кредитные карты", "кредитный лимит", "льготный период"
        ],
        "Вклад": ["депозит", "процентная ставка", "капитализация"],
        "Кредит": ["займ", "одобрение", "ежемесячный платеж"],
        "Ипотека": ["ипотека", "первоначальный взнос", "недвижимость"],
        "Денежные переводы": ["перевод", "комиссия", "получатель"],
        "Эквайринг": ["эквайринг", "pos-терминал", "оплата картой"],
    }


def load_keyword_samples(col_clusters):
    texts: List[str] = []
    labels: List[List[str]] = []
    canon_map = canonical_topic_mapping()

    for cl in col_clusters.find({}):
        raw_name = (cl.get("name") or "").strip()
        if not raw_name:
            continue
        canonical = canon_map.get(raw_name.lower(), raw_name)
        keywords = cl.get("keywords", []) or []
        for kw in keywords:
            kw_str = str(kw).strip()
            if not kw_str:
                continue
            texts.append(kw_str)
            labels.append([canonical])

    for canonical, examples in expand_synonyms_seed().items():
        for ex in examples:
            texts.append(ex)
            labels.append([canonical])

    return texts, labels


def main():
    db = get_db()
    col_clusters = db["clusters"]

    texts, labels = load_keyword_samples(col_clusters)
    if not texts:
        raise SystemExit("No training data found (clusters empty)")

    print(f"Training topics model on {len(texts)} keyword examples (canonicalized)")

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(texts, Y, test_size=0.2, random_state=42)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"))),
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nTopics Model Performance:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/topics_model.pkl")
    joblib.dump(mlb, "models/topics_mlb.pkl")
    print("Saved topics_model.pkl and topics_mlb.pkl")

    # Quick smoke test
    tests = [
        "мобильное приложение зависает",
        "очередь и обслуживание",
        "дебетовая карта и банкомат",
        "кредитная карта с лимитом",
    ]
    for t in tests:
        probs = clf.predict_proba([t])[0]
        chosen = [(mlb.classes_[i], float(p)) for i, p in enumerate(probs) if p >= 0.3]
        print(f"{t} -> {chosen}")


if __name__ == "__main__":
    main()