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
from dotenv import load_dotenv

load_dotenv()

# Отключаем предупреждения SSL только если нужно
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parser.log', encoding='utf-8'),
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

BANK_CODE = os.getenv("BANK_CODE", "gazprombank")
START_URL = f"https://www.banki.ru/services/responses/bank/{BANK_CODE}/"
AJAX_URL = "https://www.banki.ru/services/responses/list/ajax/"
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "gazprombank_reviews.json")
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "50"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

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
    "Referer": f"https://www.banki.ru/services/responses/bank/{BANK_CODE}/",
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


def save_reviews(data: list, file_path: str = OUTPUT_FILE):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def get_filters(session):
    logger.info("Получаем список доступных услуг...")

    try:
        response = session.get(START_URL, timeout=30, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        filters = {}
        all_links = soup.find_all('a', href=True)

        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            if '/services/responses/bank/gazprombank/product/' in href:
                prod = extract_product_code(href)
                if prod and text and text not in filters:
                    filters[text] = prod
                    logger.debug(f"Найден фильтр: '{text}' -> '{prod}'")

        if not filters:
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)

                if 'product=' in href:
                    prod = extract_product_code(href)
                    if prod and text:
                        filters[text] = prod
                        logger.debug(f"Найден фильтр (параметры): '{text}' -> '{prod}'")

        filters["Все"] = None

        logger.info(f"Найдено фильтров: {len(filters)}")
        for name, code in filters.items():
            logger.info(f"  - {name}: {code}")

        return filters

    except Exception as exc:
        logger.error(f"Ошибка получения фильтров: {exc}")
        return {
            "Все": None,
            "Дебетовые карты": "debitcards",
            "Кредиты": "credits",
            "Ипотека": "mortgage",
            "Вклады": "deposits"
        }


def fetch_reviews_for_service(service_name, product_code, all_reviews, session):
    logger.info(f"=== Сбор отзывов для: {service_name} ===")

    page = 1
    service_reviews_count = 0

    while True:
        params = {
            "bank": BANK_CODE,
            "page": page,
            "is_countable": "on"
        }

        if product_code:
            params["product"] = product_code

        try:
            response = session.get(AJAX_URL, params=params, timeout=60, verify=True)

            if response.status_code != 200:
                logger.warning(f"Код ответа {response.status_code} – прекращаем.")
                break

            data = response.json()
            items = data.get("data", [])

            if not items:
                logger.info(f"Отзывов больше нет для '{service_name}'.")
                break

            for item in items:
                date_str = item.get("dateCreate") or item.get("published_at") or item.get("created_at")
                review_date = None
                if date_str:
                    try:
                        if 'T' in date_str:
                            review_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            review_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        pass

                review_text = clean_html(item.get("text"))

                review_data = {
                    "source": "banki.ru",
                    "company": "Газпромбанк",
                    "service_type": service_name,
                    "text": review_text,
                    "rating": item.get("grade"),
                    "author": item.get("userName"),
                    "title": item.get("title"),
                    "date": review_date,
                    "url": f"https://www.banki.ru/services/responses/review/{item.get('id')}/" if item.get(
                        "id") else None,
                    "status": "unprocessed",
                    "parsed_at": datetime.utcnow()
                }

                all_reviews.append(review_data)
                service_reviews_count += 1

                if len(all_reviews) % BUFFER_SIZE == 0:
                    save_reviews(all_reviews)
                    logger.info(f"Буфер сохранен: {len(all_reviews)} отзывов")

            if page % 5 == 0:
                logger.info(f"[{service_name}] Страница {page}: {service_reviews_count} отзывов")

            page += 1
            time.sleep(1.5)

        except requests.exceptions.RequestException as exc:
            logger.error(f"Ошибка запроса: {exc}")
            break
        except json.JSONDecodeError as exc:
            logger.error(f"Ошибка парсинга JSON: {exc}")
            break
        except Exception as exc:
            logger.error(f"Неожиданная ошибка: {exc}")
            break

    logger.info(f"Для '{service_name}': обработано {service_reviews_count} отзывов")
    return service_reviews_count


def main():
    logger.info("Начинаем умный сбор отзывов через API...")
    session = create_session()

    try:
        filters = get_filters(session)

        all_reviews = []
        total_reviews = 0

        for service_name, product_code in filters.items():
            service_count = fetch_reviews_for_service(service_name, product_code, all_reviews, session)
            total_reviews += service_count
            logger.info(f"Завершена услуга '{service_name}': {service_count} отзывов (всего: {total_reviews})")

        save_reviews(all_reviews)

        logger.info(f"Всего собрано: {len(all_reviews)} отзывов")
        logger.info(f"Данные сохранены в {OUTPUT_FILE}")

    finally:
        session.close()


if __name__ == "__main__":
    main()