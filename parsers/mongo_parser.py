import requests
import json
import time
import logging
import os
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    bank: str = "gazprombank"
    mongo_uri: str = MONGO_URI
    db_name: str = DB_NAME
    collection_name: str = COLLECTION_NAME
    ajax_url: str = "https://www.banki.ru/services/responses/list/ajax/"
    start_url: str = f"https://www.banki.ru/services/responses/bank/gazprombank/"
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "50"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

CONFIG = Config()

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mongo_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

def parse_date(raw_date: str) -> datetime:
    if not raw_date:
        return None

    try:
        if "T" in raw_date:
            return datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        return datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
    except Exception:
        logger.warning(f"Не удалось распарсить дату: {raw_date}")
        return None

COOKIES = (
    "_flpt_sso_auth_user_in_segment=on; "
    "_flpt_percent_zone=10; "
    "_flpt_swap_deposits_credits=true; "
    "_flpt_main_page_personalization_absegment=segment_b; "
    "_ga=GA1.1.1656646479.1758664978; "
    "ga_client_id=1656646479.1758664978; "
    "tmr_lvid=830e42512cdc50729302a17b05e5d485; "
    "tmr_lvidTS=1758664978526; "
    "_ym_uid=1758664979807634781; "
    "_ym_d=1758664979; "
    "ym_client_id=1758664979807634781; "
    "_gcl_au=1.1.1612960776.1758664979; "
    "_ym_isad=2; "
    "uxs_uid=118c9640-98c9-11f0-a81d-a392dbd7e0f4; "
    "domain_sid=ygJ-D3nvOCVdZFX0Iw6JB%3A1758664980026; "
    "non_auth_user_region_id=4; "
    "BANKI_RU_USER_IDENTITY_UID=7189648582680695306; "
    "_flpt_header_mobile_app_links=hide; "
    "__lhash_=92f509eef37f98e03f290b8928d6b4bd; "
    "views_counter=%7B%22folk_rating.response%22%3A%5B11372578%5D%7D; "
    "PHPSESSID_NG_USER_RATING=baocmr6vnpuu7m3a6ume6dug8p; "
    "gtm-session-start=1758711643102; "
    "_ym_visorc=b; "
    "banki_prev_page=/services/responses/bank/gazprombank/; "
    "counter_session=3; "
    "BANKI_RU_MYBANKI_ID=40cf9ecc-ff3b-41a2-b37f-9495c5e4e52d; "
    "_banki_ru_mybanki_id_migration=2024-08-14-updatedCookieDomain; "
    "_ga_MEEKHDWY53=GS2.1.s1758711642$o3$g1$t1758712086$j60$l0$h0; "
    "tmr_detect=0%7C1758712089462; "
    "aff_sub3=/services/responses/bank/response/11372578/; "
    "_ga_EFC0FSWXRL=GS2.1.s1758711639$o3$g1$t1758712110$j36$l0$h0"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Referer": f"https://www.banki.ru/services/responses/bank/{CONFIG.bank}/",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": COOKIES,
    "Accept": "application/json",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
}

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return None
    return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)


def extract_product_code(href: str) -> str:
    if not href:
        return None

    parsed = urlparse(href)
    if parsed.query:
        qs = parse_qs(parsed.query)
        if 'product' in qs:
            return qs['product'][0]

    if '/product/' in href:
        return href.rstrip('/').split('/product/')[-1].split('/')[0]

    return None

def get_filters(session=None):
    logger.info("Находим все доступные фильтры(услуги) ")

    try:
        if session is None:
            session = requests.Session()
            session.headers.update(HEADERS)

        r = session.get(CONFIG.start_url, timeout=30, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        filters = {}
        all_links = soup.find_all('a', href=True)

        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            product_code = extract_product_code(href)
            if product_code and text and text not in filters:
                filters[text] = product_code
                logger.debug(f"Найден фильтр: '{text}' -> '{product_code}'")

        filters["Все"] = None

        logger.info(f"Найдено фильтров: {len(filters)}")
        for name, code in filters.items():
            logger.info(f"  - {name}: {code}")

        return filters

    except Exception as e:
        logger.error(f"Ошибка получения фильтров: {e}")
        return {
            "Все": None,
            "Дебетовые карты": "debitcards",
            "Кредиты": "credits",
            "Ипотека": "mortgage",
            "Вклады": "deposits"
        }


def connect_to_mongodb():
    try:
        client = MongoClient(CONFIG.mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        db = client[CONFIG.db_name]
        collection = db[CONFIG.collection_name]

        client.admin.command('ping')
        logger.info(f"Подключение к MongoDB успешно: {CONFIG.db_name}.{CONFIG.collection_name}")
        return client, collection
    except Exception as e:
        logger.error(f"Ошибка подключения к MongoDB: {e}")
        return None, None


def save_review_to_mongo(collection, review_data):
    try:
        result = collection.update_one(
            {"url": review_data["url"]},
            {"$setOnInsert": review_data},
            upsert=True
        )
        return result.upserted_id is not None
    except Exception as e:
        logger.error(f"Ошибка сохранения в MongoDB: {e}")
        return False


def create_review_data(item, service_name):
    date_str = item.get("dateCreate") or item.get("published_at") or item.get("created_at")

    return {
        "source": "banki.ru",
        "company": "Газпромбанк",
        "service_type": service_name,
        "text": clean_html(item.get("text")),
        "rating": item.get("grade"),
        "author": item.get("userName"),
        "title": item.get("title"),
        "date": parse_date(date_str),
        "url": f"https://www.banki.ru/services/responses/review/{item.get('id')}/" if item.get("id") else None,
        "status": "unprocessed",
        "parsed_at": datetime.utcnow()
    }

def fetch_reviews_for_service(service_name, product_code, collection, session=None):
    logger.info(f"Сбор отзывов для: {service_name}")

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    page = 1
    saved_to_mongo = 0
    skipped_duplicates = 0

    while True:
        params = {
            "bank": CONFIG.bank,
            "page": page,
            "is_countable": "on"
        }

        if product_code:
            params["product"] = product_code

        logger.info(f"Загружаем страницу {page} для '{service_name}'...")

        try:
            r = session.get(CONFIG.ajax_url, params=params, timeout=60, verify=False)

            if r.status_code != 200:
                logger.warning(f"Код ответа {r.status_code} – прекращаем.")
                break

            data = r.json()
            items = data.get("data", [])

            if not items:
                logger.info(f"Отзывов больше нет для '{service_name}'.")
                break

            for item in items:
                review_data = create_review_data(item, service_name)

                if collection is not None:
                    if save_review_to_mongo(collection, review_data):
                        saved_to_mongo += 1
                    else:
                        skipped_duplicates += 1

            logger.info(f"Собрано {len(items)} отзывов на странице {page}")
            page += 1
            time.sleep(1.5)

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            break
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            break

    logger.info(f"Для '{service_name}': обработано {saved_to_mongo + skipped_duplicates} отзывов")
    if collection is not None:
        logger.info(f"Сохранено новых: {saved_to_mongo}, пропущено дубликатов: {skipped_duplicates}")

    return saved_to_mongo, skipped_duplicates


def main():
    logger.info("Сбор отзывов через API в MongoDB")
    session = create_session()

    try:
        filters = get_filters(session)
        client, collection = connect_to_mongodb()

        if collection is None:
            logger.error("Не удалось подключиться к MongoDB. Завершение работу.")
            return

        total_saved = 0
        total_skipped = 0

        for service_name, product_code in filters.items():
            saved, skipped = fetch_reviews_for_service(service_name, product_code, collection, session)
            total_saved += saved
            total_skipped += skipped

        logger.info(f"Всего обработано: {total_saved + total_skipped} отзывов")
        logger.info(f"Сохранено новых: {total_saved}")
        logger.info(f"Пропущено дубликатов: {total_skipped}")
        logger.info(f"Данные сохранены в MongoDB: {CONFIG.db_name}.{CONFIG.collection_name}")

    finally:
        session.close()
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()