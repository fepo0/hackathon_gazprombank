import os
import sys

from datetime import datetime
from dateutil import tz
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from PyQt5 import QtWidgets, QtCore

# Авторизация
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
try:
    client.admin.command("ping")
    print("Подключение к Mongo")
except Exception as e:
    print("Ошибка подключения: ", e)
    sys.exit(1)

db = client["banki_db"]

col_raw = db["reviews_raw"]
col_labeled = db["reviews_labeled"]
col_topics = db["topics"]
col_sentiments = db["sentiments"]
col_clusters = db["clusters"]

def now_iso():
    return datetime.utcnow()

def take_next_unprocessed():
    doc = col_raw.find_one_and_update(
        {"status": "unprocessed"},
        {"$set": {"status": "in_processed",
                  "in_progress_by": user,
                  "in_progress_at": now_iso()
                  }},
    )
    return doc

def mark_raw_processed(raw_id):
    col_raw.update_one(
        {"id": raw_id},
        {"$set": {"status": "processed",
                  "processed_at": now_iso()
                  }}
    )

def ensure_topics_and_sentiments_loaded():
    topics = list(col_topics.find({}))
    sentiments = list(col_sentiments.find({}))
    if not topics:
        print("Коллекция topics пустая")
    if not sentiments:
        print("Коллекция sentiments пустая")
    return topics, sentiments

def save_labeled(raw_doc, selected_topics_names, selected_sentiments_labels):
    assert len(selected_topics_names) == len(selected_sentiments_labels)
    labeled = {
        "raw_review_id": raw_doc["id"],
        "text": raw_doc.get("text", ""),
        "title": raw_doc.get("title", ""),
        "topics": selected_topics_names,
        "sentiments": selected_sentiments_labels,
        "labeled_by": user,
        "labeled_at": now_iso(),
        "status": "draft"
    }
    res = col_labeled.insert_one(labeled)
    labeled_id = res.inserted_id
    print(f"Сохранен обработанный отзыв {labeled_id}")

    for topic_name in selected_topics_names:
        topic_doc = col_topics.find_one({"name": topic_name})
        cluster_update = False

        if topic_doc:
            t_cluster_id = topic_doc.get("cluster_id")
            if t_cluster_id is not None:
                upd = col_clusters.update_one(
                    {"cluster_id": t_cluster_id},
                    {"$addToSet": {"example_reviews": labeled_id}},
                )
                if upd.modified_count > 0 or upd.matched_count > 0:
                    cluster_update = True

        if not cluster_update:
            upd = col_clusters.update_one(
                {"name": {"$regex": f"^{topic_name}$",
                          "$options": "i"}},
                {"$addToSet": {"example_reviews": labeled_id}},
            )
            if upd.modified_count > 0 or upd.matched_count > 0:
                cluster_update = True

        if not cluster_update:
            upd = col_clusters.update_one(
                {"keywords": {"$elemMatch": {"$regex": f"^{topic_name}$",
                                             "$options": "i"}}},
                {"$addToSet": {"example_reviews": labeled_id}},
            )
            if upd.modified_count > 0 or upd.matched_count > 0:
                cluster_update = True

        if not cluster_update and topic_doc:
            try:
                tid_str = str(topic_doc["_id"])
                upd = col_topics.update_one(
                    {"cluster_id": tid_str},
                    {"$addToSet": {"example_reviews": labeled_id}},
                )
                if upd.modified_count > 0 or upd.matched_count > 0:
                    cluster_update = True
            except Exception as e:
                pass

        if cluster_update:
            print(f"Добавлен labeled_id в кластер для услуги(товара) {topic_name}")
        else:
            print(f"Не найдено подходящего кластера для темы {topic_name}")

    return labeled_id

# GUI
class LabelinngWindow(QtWidgets.QWidget):
    def __init__(self, raw_doc, topics_list, sentiments_list, prefill=None):
        """
        :param raw_doc: кейс из reviews_raw
        :param topics_list: список topics
        :param sentiments_list: список sentiments
        :param prefill: проверка верификации
        """
        super().__init__()
        self.raw = raw_doc
        self.topics_list = topics_list
        self.sentiments_list = sentiments_list
        self.prefill = prefill or {}
        self.selected_order = []
        self.topic_checkboxes = {}
        self.topic_sentiments = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Обработка")
        layout = QtWidgets.QVBoxLayout()

        # Заголовок
        title_label = QtWidgets.QLabel(f"<b>{self.raw.get('title', '(Нет заголовка)')}</b>")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Текст + скролл
        text_area = QtWidgets.QPlainTextEdit()
        text_area.setPlainText(self.raw.get("text", ""))
        text_area.setReadOnly(True)
        text_area.setFixedHeight(200)
        layout.addWidget(text_area)

        # Услуги (товары) + чекбоксы
        NUM_COLS = 3

        topics_groupbox = QtWidgets.QGroupBox("Услуги (товары), выберите те, что относятся к отзыву")
        topics_layout = QtWidgets.QGridLayout()

        for i, tdoc in enumerate(self.topics_list):
            tname = tdoc.get("name")
            cb = QtWidgets.QCheckBox(tname)
            cb.stateChanged.connect(lambda state, name=tname: self.on_topic_toggled(name, state))
            row = i // NUM_COLS
            col = i % NUM_COLS
            topics_layout.addWidget(cb, row, col)
            self.topic_checkboxes[tname] = cb

        topics_groupbox.setLayout(topics_layout)
        layout.addWidget(topics_groupbox)

        # Область выбора
        sel_groupbox = QtWidgets.QGroupBox("Выбранные услуги (товары) и тональности")
        self.sel_layout = QtWidgets.QVBoxLayout()
        sel_groupbox.setLayout(self.sel_layout)
        layout.addWidget(sel_groupbox)

        # Предварительное заполнение (по возможности)
        if self.prefill.get("topics"):
            for name, sentiment in zip(self.prefill.get("topics", []), self.prefill.get("sentiments", [])):
                # После галочек настроения
                if name in self.topic_checkboxes:
                    self.topic_checkboxes[name].setChecked(True)
                QtCore.QTimer.singleShot(100, lambda n=name, s=sentiment: self.set_prefill_sentiment(n, s))

        # Кнопка сохранения
        bth_save = QtWidgets.QPushButton("Сохранить")
        bth_save.clicked.connect(self.on_save)
        layout.addWidget(bth_save)

        self.setLayout(layout)
        self.resize(800, 600)

    def on_topic_toggled(self, name, state):
        if state == 2:
            if name not in self.selected_order:
                self.selected_order.append(name)
                self.add_selected_row(name)
        else:
            if name in self.selected_order:
                self.selected_order.remove(name)
                self.remove_selected_row(name)

    def add_selected_row(self, name):
        # + Горизонтальный блок
        hbox = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(name)
        lbl.setFixedWidth(300)
        hbox.addWidget(lbl)

        # Настроение + радиокнопки
        bg = QtWidgets.QButtonGroup(self)
        radios = []
        for sdoc in self.sentiments_list:
            rb = QtWidgets.QRadioButton(sdoc["label"])
            bg.addButton(rb)
            hbox.addWidget(rb)
            radios.append(rb)

        # По умолчанию нейтрально
        for rb in radios:
            if rb.text().lower() in ["нейтрально", "нейтральный"]:
                rb.setChecked(True)
                break

        # Контейнер
        container = QtWidgets.QWidget()
        container.setLayout(hbox)
        container.setObjectName(f"sel_row_{name}")
        self.sel_layout.addWidget(container)

        self.topic_sentiments[name] = bg

    def remove_selected_row(self, name):
        # Удалить контейнер
        for i in range(self.sel_layout.count()):
            w = self.sel_layout.itemAt(i).widget()
            if w and w.objectName() == f"sel_row_{name}":
                self.sel_layout.takeAt(i)
                w.deleteLater()
                break
        if name in self.topic_sentiments:
            del self.topic_sentiments[name]

    def set_prefill_sentiment(self, name, sentiment_label):
        # Поставить нужный радиобатон
        bg = self.topic_sentiments.get(name)
        if not bg:
            return
        for btn in bg.buttons():
            if btn.text().strip().lower() == sentiment_label.strip().lower():
                btn.setChecked(True)
                return

    def coolect_result(self):
        topics = list(self.topics_order)
        sentiments = []
        for t in topics:
            bg = self.topic_sentiments.get(t)
            choice = None
            if bg:
                for bth in bg.buttons():
                    if bth.isChecked():
                        choice = bth.text()
                        break
            sentiments.append(choice or "")
        return topics, sentiments

    def on_save(self):
        topics, sentiments = self.coolect_result()
        if not topics:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите хотя бы 1 услугу (товар)")
            return
        if any(s == "" for s in sentiments):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите тональность для каждой услуги (тоавра)")
            return

        try:
            labeled_id = save_labeled(self.raw, topics, sentiments)
            col_raw.update_one(
                {"_id": self.raw["id"]},
                {"$set": {"last_labeled_id": labeled_id}}
            )
            QtWidgets.QMessageBox.information(self, "Сохранено", f"Обработка сохранена (id {labeled_id})")
            self.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

# Запуск
def process_all_unprocessed_reviews():
    topics_list, sentiments_list = ensure_topics_and_sentiments_loaded()
    if not topics_list or not sentiments_list:
        print("Отсуствуют topics или sentiments")
        return

    app = QtWidgets.QApplication(sys.argv)

    while True:
        raw_doc = take_next_unprocessed()
        if not raw_doc:
            print("Нет unprocessed отзывов")
            break
        print(f"Обработка: {raw_doc['_id']}")

        win = LabelinngWindow(raw_doc, topics_list, sentiments_list)
        win.show()
        app.exec_()

        lr = col_raw.find_one({"_id": raw_doc["id"]})
        if lr.get("last_labeled_id"):
            print(f"Смотрим {raw_doc['id']} labeled: {lr['last_labeled_id']}")
        else:
            col_raw.update_one({"_id": raw_doc["id"]},
                               {"$set": {"status": "unprocessed"},
                                "$unset": {"in_progress_by": "",
                                           "in_progress_at": ""}
                                })
            print(f"Пользователь пропустил отзыв {raw_doc['id']}; Возращаем в unprocessed")

    while True:
        to_verify = col_labeled.find_one({"status": "draft",
                                          "labeled_by": {"$ne": user}})
        if not to_verify:
            print("Нет draft для верификации")
            break
        print(f"Проверка верификации: {to_verify['id']} (к {to_verify['labeled_by']})")
        raw_doc = col_raw.find_one({"_id": to_verify["raw_review_id"]})
        if not raw_doc:
            print("Не найден соответствующий raw_doc, помечаем orphan")
            col_labeled.update_one({"_id": to_verify["_id"]},
                                   {"$set": {"status": "orphan",}})
            continue

        prefill = {"topics": to_verify.get("topics", []),
                   "sentiments": to_verify.get("sentiments", [])}

        win = LabelinngWindow(raw_doc, topics_list, sentiments_list, prefill=prefill)
        win.show()
        app.exec_()

        col_labeled.update_one({"_id": to_verify["_id"]},
                               {"$set": {"status": "verified",
                                        "verified_by": user,
                                         "verified_at": now_iso()}})
        print(f"Помечаем {to_verify['id']} отмечено как verified {user}")

        col_raw.update_one({"_id": raw_doc["_id"]},
                           {"$set": {"status": "processed",
                                     "processed_at": now_iso()}})

    print("Готово. Все кейсы обработаны и проверены")

class UpdateLabelinngWindow(LabelinngWindow):
    def __init__(self, raw_doc, topics_list, sentiments_list, labeled_doc):
        super().__init__(raw_doc,
                         topics_list,
                         sentiments_list,
                         prefill={"topics": labeled_doc.get("topics", []),
                                  "sentiments": labeled_doc.get("sentiments", [])})
        self.labeled_doc = labeled_doc

    def on_save(self):
        topics, sentiments = self.coolect_result()
        if not topics:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите хотя бы 1 услугу (товар)")
            return
        if any(s == " " for s in sentiments):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите тональность для каждой услуги (товара)")
            return

        try:
            col_labeled.update_one(
                {"_id": self.labeled_doc["id"]},
                {"$set":{
                    "topics": sentiments,
                    "sentiments": sentiments,
                    "labeled_by": self.labeled_doc.get("labeled_by", ""),
                    "labeled_at": self.labeled_doc.get("labeled_at", now_iso()),
                    "status": "verified",
                    "verified_by": user,
                    "verified_at": now_iso()
                }}
            )
            labeled_id = self.labeled_doc["_id"]
            for topic_name in topics:
                topic_doc = col_topics.find_one({"name": topic_name})
                cluster_updated = False
                if topic_doc:
                    t_cluster_id = topic_doc.get("cluster_id")
                    if t_cluster_id is not None:
                        col_clusters.update_one({"cluster_id": t_cluster_id},
                                                {"$addToSet": {"example_reviews": labeled_id}})
                        cluster_updated = True
                    if not cluster_updated:
                        col_clusters.update_one({"name": {"$regex": f"^{topic_name}$", "$options": "i"}},
                                                {"$addToSet": {"example_reviews": labeled_id}})
            col_raw.update_one({"_id": self.raw["_id"]},
                               {"$set": {"status": "processed",
                                         "processed_at": now_iso()}})
            QtWidgets.QMessageBox.information(self, "Сохранено", f"Обновлено (id: {labeled_id})")
            self.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось обновить: {e}")

def process_verification_phase():
    topics_list, sentiments_list = ensure_topics_and_sentiments_loaded()
    if not topics_list or not sentiments_list:
        return
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    while True:
        to_verify = col_labeled.insert_one({"status": "draft",
                                            "labeled_by": {"$ne": user}})
        if not to_verify:
            print("Нет draft для верификации")
            break
        raw_doc = col_labeled.find_one({"_id": to_verify["raw_review_id"]})
        if not raw_doc:
            col_labeled.update_one({"_id": to_verify["_id"]},
                                   {"$set": {"status": "orphan"}})
            continue

        win = UpdateLabelinngWindow(raw_doc, topics_list, sentiments_list, to_verify)
        win.show()
        app.exec_()

# Запуск
if __name__ == "__main__":
    process_all_unprocessed_reviews()
    process_verification_phase()
    print("Все завершено")