import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import re
import joblib
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
try:
    from transformers import pipeline
except Exception:
    pipeline = None
try:
    from razdel import sentenize
except Exception:
    sentenize = None

MODELS_DIR = "models"
TOPICS_MODEL_PATH = os.path.join(MODELS_DIR, "topics_model.pkl")
TOPICS_MLB_PATH = os.path.join(MODELS_DIR, "topics_mlb.pkl")
SARC_MODEL_ID = os.environ.get("RU_SARC_MODEL", "MaryObr/sarcasm-RuBERT")
SENT_MODEL_ID = os.environ.get("RU_SENT_MODEL", "blanchefort/rubert-base-cased-sentiment")

THRESH = 0.3

def load_mongo_uri_from_file(username_prompt=True):
    path = "autorization.txt"
    if os.path.exists(path):
        txt = open(path, "r", encoding="utf-8").read().strip()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if not lines:
            return None
    return None

MONGO_URI = os.environ.get("MONGO_URI") or "mongodb://localhost:27017/"

topics_model = None
topics_mlb = None
sent_model = None
sent_label_encoder = None
sarcasm_pipe = None
sentiment_pipe = None

if os.path.exists(TOPICS_MODEL_PATH) and os.path.exists(TOPICS_MLB_PATH):
    try:
        topics_model = joblib.load(TOPICS_MODEL_PATH)
        topics_mlb = joblib.load(TOPICS_MLB_PATH)
        print("Loaded topics_model and topics_mlb")
    except Exception as e:
        print("Error loading topics model:", e)
else:
    print("Topics model files not found in", MODELS_DIR)

sent_model = None
sent_label_encoder = None

if pipeline is not None:
    try:
        sarcasm_pipe = pipeline(
            "text-classification",
            model=SARC_MODEL_ID,
            tokenizer=SARC_MODEL_ID,
            top_k=None,
            truncation=True,
            framework="pt",
            device=-1,
        )
        print(f"Loaded sarcasm pipeline: {SARC_MODEL_ID}")
    except Exception as e:
        print("Sarcasm pipeline not available:", e)
        sarcasm_pipe = None
    try:
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=SENT_MODEL_ID,
            tokenizer=SENT_MODEL_ID,
            top_k=None,
            truncation=True,
            framework="pt",
            device=-1,
        )
        print(f"Loaded sentiment pipeline: {SENT_MODEL_ID}")
    except Exception as e:
        print("Sentiment pipeline not available:", e)
        sentiment_pipe = None

client = MongoClient(MONGO_URI)
db = client.get_database("banki_db")
col_reviews_raw = db["reviews_raw"]
col_reviews_labeled = db["reviews_labeled"]
col_topics = db["topics"]
col_sentiments = db["sentiments"]
col_clusters = db["clusters"]
col_predictions = db.get_collection("predictions")
col_otz = db.get_collection("otz")

app = FastAPI(title="Gazprombank Reviews API & Dashboard")

class SingleReview(BaseModel):
    id: int
    text: str
    date: Optional[str] = None

class PredictRequest(BaseModel):
    data: List[SingleReview]

def predict_topics_for_text(text: str) -> List[str]:
    if topics_model is None or topics_mlb is None:
        return []
    try:
        if hasattr(topics_model, "predict_proba"):
            probs = topics_model.predict_proba([text])
            if isinstance(probs, list):
                probs_arr = probs[0]
            else:
                probs_arr = probs
            arr = probs_arr[0] if probs_arr.ndim == 2 else probs_arr
            chosen = []
            for i, p in enumerate(arr):
                if p >= THRESH:
                    chosen.append(topics_mlb.classes_[i])
            if not chosen:
                preds = topics_model.predict([text])
                if preds is not None and len(preds) > 0:
                    binvec = preds[0]
                    for i, v in enumerate(binvec):
                        if v == 1:
                            chosen.append(topics_mlb.classes_[i])
                if not chosen:
                    top_idx = int(arr.argmax())
                    chosen.append(topics_mlb.classes_[top_idx])
            return chosen
        else:
            preds = topics_model.predict([text])
            binvec = preds[0]
            chosen = [topics_mlb.classes_[i] for i, v in enumerate(binvec) if v == 1]
            if not chosen:
                return []
            return chosen
    except Exception:
        try:
            preds = topics_model.predict([text])
            binvec = preds[0]
            chosen = [topics_mlb.classes_[i] for i, v in enumerate(binvec) if v == 1]
            return chosen
        except Exception:
            return []


def build_topic_keywords_map() -> dict:
    topic_to_keys = {}
    try:
        clusters = list(col_clusters.find({}))
        for cl in clusters:
            name = (cl.get("name") or "").strip()
            if not name:
                continue
            keys = cl.get("keywords", []) or []
            keys_norm = set()
            for kw in keys:
                s = str(kw).strip().lower()
                if s:
                    keys_norm.add(s)
            keys_norm.add(name.lower())
            topic_to_keys[name] = list(keys_norm)
    except Exception:
        topic_to_keys = {}
    return topic_to_keys


def split_by_contrast(sentence: str) -> list:
    parts = re.split(r"\b(но|однако|зато|при этом)\b", sentence, flags=re.IGNORECASE)
    chunks = []
    buf = []
    for p in parts:
        if re.fullmatch(r"(?i)(но|однако|зато|при этом)", p or ""):
            if buf:
                chunks.append("".join(buf).strip())
                buf = []
        else:
            buf.append(p)
    if buf:
        chunks.append("".join(buf).strip())
    return [c for c in chunks if c]


def extract_topic_contexts(text: str, topic: str, keywords: list) -> list:
    contexts = []
    if sentenize is not None:
        sentences = [s.text for s in sentenize(text)]
    else:
        sentences = [s for s in re.split(r"[\.!?\n]+", text) if s.strip()]
    keys = [k.lower() for k in (keywords or [])]
    keys.append(topic.lower())
    for sent in sentences:
        low = sent.lower()
        if any(k in low for k in keys):
            for clause in split_by_contrast(sent):
                cl = clause.strip()
                if any(k in cl.lower() for k in keys):
                    contexts.append(cl)
    if not contexts:
        contexts = [text]
    return contexts

def predict_sentiment_for_text_topic(text: str, topic: str) -> str:
    t2k = build_topic_keywords_map()
    keys = t2k.get(topic, [])
    contexts = extract_topic_contexts(text, topic, keys)

    votes = []
    for ctx in contexts:
        label = None
        is_sarcasm = False
        if sarcasm_pipe is not None:
            try:
                sres = sarcasm_pipe(ctx)
                if isinstance(sres, list) and len(sres) > 0:
                    stop = max(sres, key=lambda x: x.get("score", 0.0))
                    slabel = (stop.get("label") or "").lower()
                    if any(k in slabel for k in ["sarcasm", "ирони"]):
                        is_sarcasm = True
            except Exception:
                is_sarcasm = False

        if sentiment_pipe is not None:
            try:
                r = sentiment_pipe(ctx)
                if isinstance(r, list) and len(r) > 0:
                    top = max(r, key=lambda x: x.get("score", 0.0))
                    lbl = (top.get("label") or "").lower()
                    if "pos" in lbl:
                        label = "положительно"
                    elif "neg" in lbl:
                        label = "отрицательно"
                    else:
                        label = "нейтрально"
            except Exception:
                label = None

        if not label:
            low = ctx.lower()
            negative_cues = ["зависает", "тормозит", "лагает", "не работает", "ошибка", "сбой", "проблема", "медленно", "долго", "неудобно", "не нравится"]
            positive_cues = ["быстро", "работает", "удобно", "понравилось", "отлично", "хорошо", "стабильно", "рекомендую"]
            if any(c in low for c in negative_cues):
                label = "отрицательно"
            elif any(c in low for c in positive_cues):
                label = "положительно"
            else:
                label = "нейтрально"

        if is_sarcasm and label == "положительно":
            label = "отрицательно"
        votes.append(label)

    if not votes:
        return "нейтрально"
    pos = sum(1 for v in votes if v == "положительно")
    neg = sum(1 for v in votes if v == "отрицательно")
    if pos > neg:
        return "положительно"
    if neg > pos:
        return "отрицательно"
    return "нейтрально"

@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    if topics_model is None or topics_mlb is None:
        raise HTTPException(status_code=503, detail="Topic models are not loaded. Ensure models/topics_*.pkl exist.")
    items = payload.data
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="Invalid payload")
    out = {"predictions": []}
    for rec in items:
        text = rec.text
        rid = rec.id
        topics_pred = predict_topics_for_text(text)
        sentiments_pred = []
        for tp in topics_pred:
            s = predict_sentiment_for_text_topic(text, tp)
            sentiments_pred.append(s if s else "нейтрально")
        out["predictions"].append({"id": rid, "topics": topics_pred, "sentiments": sentiments_pred})
        try:
            provided_date = None
            if getattr(rec, "date", None):
                try:
                    dstr = rec.date.strip()
                    if len(dstr) == 10:
                        provided_date = datetime.strptime(dstr, "%Y-%m-%d")
                    else:
                        provided_date = datetime.fromisoformat(dstr.replace("Z", "+00:00"))
                except Exception:
                    provided_date = None

            col_predictions.insert_one({
                "review_id": rid,
                "text": text,
                "topics_pred": topics_pred,
                "sentiments_pred": sentiments_pred,
                "model_version": {
                    "topics": os.path.basename(TOPICS_MODEL_PATH) if os.path.exists(TOPICS_MODEL_PATH) else None,
                    "sentiment": "sarcasm_pipe" if sarcasm_pipe is not None else "heuristics"
                },
                "predicted_at": datetime.utcnow()
            })
            col_otz.insert_one({
                "source": "api",
                "external_id": rid,
                "text": text,
                "topics": topics_pred,
                "sentiments": sentiments_pred,
                "date": provided_date or datetime.utcnow(),
                "inserted_at": datetime.utcnow()
            })
        except Exception:
            pass
    return JSONResponse(content=out)

@app.get("/api/topics")
async def api_topics():
    try:
        pipeline = [
            {"$unwind": "$topics"},
            {"$group": {"_id": "$topics", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        items = list(col_otz.aggregate(pipeline))
        docs = [{"id": it["_id"], "name": it["_id"], "count": it["count"]} for it in items]
    except Exception:
        docs = []
    return JSONResponse(content={"topics": docs})

@app.get("/api/sentiments")
async def api_sentiments():
    labels = [
        {"id": "положительно", "label": "положительно"},
        {"id": "нейтрально", "label": "нейтрально"},
        {"id": "отрицательно", "label": "отрицательно"},
    ]
    return JSONResponse(content={"sentiments": labels})

@app.get("/api/stats")
async def api_stats(topic: Optional[str] = None, period_months: int = 18, start_date: Optional[str] = None, end_date: Optional[str] = None):
    date_match = {}
    try:
        if start_date:
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            date_match.setdefault("$gte", sd)
        if end_date:
            ed = datetime.strptime(end_date, "%Y-%m-%d")
            date_match.setdefault("$lte", ed.replace(hour=23, minute=59, second=59, microsecond=999999))
    except Exception:
        date_match = {}

    base_match = {}
    if date_match:
        base_match["date"] = date_match

    try:
        tpipe = [
            {"$match": base_match} if base_match else {"$match": {}},
            {"$unwind": "$topics"},
            {"$group": {"_id": "$topics", "count": {"$sum": 1}}}
        ]
        topics_list = [it["_id"] for it in col_otz.aggregate(tpipe)]
    except Exception:
        topics_list = []

    stats = {}
    for tname in topics_list:
        pipeline = []
        match_stage = {"topics": tname}
        if base_match:
            match_stage.update(base_match)
        pipeline.append({"$match": match_stage})
        pipeline.append({"$unwind": {"path": "$topics", "includeArrayIndex": "idx"}})
        pipeline.append({"$match": {"topics": tname}})
        pipeline.append({"$addFields": {"sent_at_idx": {"$arrayElemAt": ["$sentiments", "$idx"]}}})
        pipeline.append({"$group": {"_id": "$sent_at_idx", "count": {"$sum": 1}}})

        try:
            total = col_otz.count_documents(match_stage)
            res = list(col_otz.aggregate(pipeline))
            dist = {r["_id"]: r["count"] for r in res}
        except Exception:
            total = 0
            dist = {}
        stats[tname] = {"total": total, "dist": dist}

    timeseries = []
    if topic:
        months = []
        if start_date and end_date:
            try:
                sd = datetime.strptime(start_date, "%Y-%m-%d")
                ed = datetime.strptime(end_date, "%Y-%m-%d")
                cur = datetime(sd.year, sd.month, 1)
                endm = datetime(ed.year, ed.month, 1)
                while cur <= endm:
                    months.append((cur.year, cur.month))
                    if cur.month == 12:
                        cur = datetime(cur.year + 1, 1, 1)
                    else:
                        cur = datetime(cur.year, cur.month + 1, 1)
            except Exception:
                months = []
        if not months:
            now = datetime.utcnow()
            base = datetime(now.year, now.month, 1)
            for i in range(period_months-1, -1, -1):
                y = base.year
                m = base.month - i
                while m <= 0:
                    m += 12
                    y -= 1
                months.append((y, m))

        ts = []
        for (y, m) in months:
            start = datetime(y, m, 1)
            if m == 12:
                end = datetime(y+1, 1, 1)
            else:
                end = datetime(y, m+1, 1)
            match = {"topics": topic, "date": {"$gte": start, "$lt": end}}
            count = col_otz.count_documents(match)
            ts.append({"year": y, "month": m, "count": count})
        timeseries = ts

    return JSONResponse(content={"stats": stats, "timeseries": timeseries})


DASH_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gazprombank Reviews Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body{font-family: Arial, sans-serif; padding:16px; max-width:1200px; margin:auto;}
    .row{display:flex; gap:20px; align-items:flex-start;}
    .card{background:#fff; border:1px solid #eee; padding:12px; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.04);}
    .left{flex:1;}
    .right{width:480px;}
    select, button{padding:6px 8px; margin-right:8px;}
    h2{margin:6px 0 12px 0;}
    ul.topics{list-style:none;padding:0; max-height:320px; overflow:auto;}
    ul.topics li{padding:6px; border-bottom:1px solid #f0f0f0;}
  </style>
</head>
<body>
  <h1>Gazprombank — Reviews Dashboard</h1>
  <div class="card">
    <div style="margin-bottom:10px;">
      <label>Выберите тему: </label>
      <select id="topicSelect"></select>
      <label style="margin-left:12px;">С даты:</label>
      <input type="date" id="startDate" />
      <label>по</label>
      <input type="date" id="endDate" />
      <button onclick="refresh()">Обновить</button>
    </div>
    <div class="row">
      <div class="left card">
        <h2>Сводка по теме</h2>
        <canvas id="sentChart" height="220"></canvas>
        <h2>Топ тем (все)</h2>
        <ul id="topicsList" class="topics"></ul>
      </div>
      <div class="right card">
        <h2>Динамика количества (по месяцам)</h2>
        <canvas id="timeChart" height="220"></canvas>
      </div>
    </div>
  </div>

<script>
async function getJSON(url){ const r = await fetch(url); return r.json(); }

let globalStats = null;

async function init(){
  const t = await getJSON('/api/topics');
  const s = await getJSON('/api/sentiments');
  const stats = await getJSON('/api/stats');
  globalStats = stats.stats;
  const topicSelect = document.getElementById('topicSelect');
  topicSelect.innerHTML = '';
  const names = Object.keys(globalStats).sort((a,b)=> (globalStats[b]?.total||0) - (globalStats[a]?.total||0));
  names.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n;
    opt.text = `${n} (${globalStats[n]?.total||0})`;
    topicSelect.appendChild(opt);
  });
  populateTopicsList(names);
  topicSelect.onchange = refresh;
  // initial render
  refresh();
}

function populateTopicsList(names){
  const ul = document.getElementById('topicsList');
  ul.innerHTML = '';
  names.forEach(n=>{
    const li = document.createElement('li');
    const d = globalStats[n] || {total:0, dist:{}};
    li.innerHTML = `<strong>${n}</strong> — ${d.total} отзывов<br><small>${JSON.stringify(d.dist)}</small>`;
    ul.appendChild(li);
  });
}

let sentChart = null, timeChart = null;

function renderSentChart(topicName){
  const stat = globalStats[topicName] || {dist:{}};
  const dist = stat.dist;
  const labels = Object.keys(dist);
  const values = labels.map(l => dist[l]);
  if(!sentChart){
    const ctx = document.getElementById('sentChart').getContext('2d');
    sentChart = new Chart(ctx, {
      type: 'bar',
      data: { labels: labels, datasets: [{ label: 'Количество', data: values }] },
      options: {}
    });
  } else {
    sentChart.data.labels = labels;
    sentChart.data.datasets[0].data = values;
    sentChart.update();
  }
}

async function renderTimeSeries(topicName){
  const sd = document.getElementById('startDate').value;
  const ed = document.getElementById('endDate').value;
  const q = new URLSearchParams({ topic: topicName, period_months: '18' });
  if (sd) q.set('start_date', sd);
  if (ed) q.set('end_date', ed);
  const resp = await getJSON(`/api/stats?${q.toString()}`);
  const ts = resp.timeseries || [];
  const labels = ts.map(x=> `${x.year}-${String(x.month).padStart(2,'0')}`);
  const vals = ts.map(x=> x.count);
  if(!timeChart){
    const ctx = document.getElementById('timeChart').getContext('2d');
    timeChart = new Chart(ctx, {
      type: 'line',
      data: { labels: labels, datasets: [{ label: 'Кол-во отзывов', data: vals, fill:false }]},
      options: {}
    });
  } else {
    timeChart.data.labels = labels;
    timeChart.data.datasets[0].data = vals;
    timeChart.update();
  }
}

function refresh(){
  const topic = document.getElementById('topicSelect').value;
  if(!topic) return;
  renderSentChart(topic);
  renderTimeSeries(topic);
}

window.onload = init;
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse(content=DASH_HTML, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
