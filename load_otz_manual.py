import os
from datetime import datetime
from typing import List, Dict

from pymongo import MongoClient


def get_mongo_uri() -> str:
    uri = os.environ.get("MONGO_URI")
    if uri:
        return uri
    return "mongodb://localhost:27017/"


def insert_manual_docs(docs: List[Dict]):
    client = MongoClient(get_mongo_uri())
    db = client.get_database(os.environ.get("DB_NAME", "banki_db"))
    col = db.get_collection("otz")

    now = datetime.utcnow()
    prepared = []
    for d in docs:
        ddate = d.get("date")
        if isinstance(ddate, str):
            try:
                if len(ddate.strip()) == 10:
                    ddate = datetime.strptime(ddate.strip(), "%Y-%m-%d")
                else:
                    ddate = datetime.fromisoformat(ddate.strip().replace("Z", "+00:00"))
            except Exception:
                ddate = None
        prepared.append({
            "source": d.get("source", "manual"),
            "external_id": d.get("id"),
            "text": d.get("text", ""),
            "topics": d.get("topics", []),
            "sentiments": d.get("sentiments", []),
            "date": ddate or now,
            "inserted_at": now,
        })

    if prepared:
        col.insert_many(prepared)
        print(f"Inserted: {len(prepared)} documents into otz")
    else:
        print("Nothing to insert")


if __name__ == "__main__":
    sample = [
        {
            "id": 1,
            "text": "Очень понравилось обслуживание в отделении, но мобильное приложение часто зависает.",
            "topics": ["Обслуживание", "Мобильное приложение"],
            "sentiments": ["положительно", "отрицательно"],
            "date": "2024-01-15"
        },
        {
            "id": 2,
            "text": "Кредитную карту одобрили быстро, но лимит слишком маленький.",
            "topics": ["Кредитная карта"],
            "sentiments": ["нейтрально"],
            "date": "2024-02-10"
        },
    ]

    insert_manual_docs(sample)