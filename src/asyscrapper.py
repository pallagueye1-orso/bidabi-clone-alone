import asyncio
import aiohttp
import csv
import os
from aiohttp import ClientSession, ClientTimeout

API_URL = "https://world.openfoodfacts.org/api/v2/search"
HEADERS = {"User-Agent": "MyAwesomeApp/1.0"}

OUTPUT_DIR = "data"

CATEGORY = "sugars" #"bread", "milk", "champagnes", "butter" 
TARGET_COUNT = 180
PAGE_SIZE = 200
MAX_PAGES = 50

MAX_CONCURRENT_REQUESTS = 10
MAX_CONCURRENT_IMAGES = 10


# -------------------------
# Helpers
# -------------------------
def get_best_image(product):
    return (
        product.get("image_url")
        or product.get("image_front_url")
        or product.get("image_small_url")
        or product.get("image_thumb_url")
    )


def is_valid_product(product):
    required = ["_id", "product_name", "categories_tags"]
    if not all(product.get(f) for f in required):
        return False
    return bool(get_best_image(product))


def extract_product_info(product):
    return [
        product.get("_id"),
        product.get("product_name"),
        ", ".join(product.get("categories_tags", [])),
        product.get("ingredients_text", ""),
        get_best_image(product)
    ]


# -------------------------
# Async API fetch
# -------------------------
async def fetch_page(session, category, page, page_size, sem):
    params = {
        "categories_tags_en": category,
        "page": page,
        "page_size": page_size
    }

    async with sem:
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with session.get(API_URL, params=params) as resp:
                    if resp.status != 200:
                        print(f"⚠ Erreur HTTP {resp.status} pour page {page}, tentative {attempt+1}")
                        if attempt < 2:
                            await asyncio.sleep(1)  # Wait 1 second before retry
                            continue
                        return []
                    data = await resp.json()
                    return data.get("products", [])
            except Exception as e:
                print(f"⚠ Erreur API page {page}, tentative {attempt+1} :", e)
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return []
        return []


# -------------------------
# Async image download
# -------------------------
async def download_image(session, url, image_id, sem, folder):
    if not url:
        return

    os.makedirs(folder, exist_ok=True)

    ext = url.split(".")[-1].split("?")[0]
    filename = os.path.join(folder, f"{image_id}.{ext}")

    if os.path.exists(filename):
        return

    async with sem:
        try:
            async with session.get(url) as resp:
                content = await resp.read()
                with open(filename, "wb") as f:
                    f.write(content)
        except Exception as e:
            print(f"⚠ Impossible de télécharger {url} :", e)


# -------------------------
# Main scraping logic
# -------------------------
async def scrape(category, target_count, page_size, max_pages):
    timeout = ClientTimeout(total=60)
    sem_api = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    sem_img = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

    folder = f"data/raw/images/{category.replace(':', '-')}"

    async with ClientSession(headers=HEADERS, timeout=timeout) as session:
        valid_products = []
        image_tasks = []
        page = 1
        consecutive_empty = 0

        while len(valid_products) < target_count and page <= max_pages:
            print(f"→ Téléchargement page {page}…")

            products = await fetch_page(session, category, page, page_size, sem_api)
            if not products:
                consecutive_empty += 1
                print(f"Aucune produit trouvé sur cette page. Consecutive empty: {consecutive_empty}")
                if consecutive_empty >= 3:
                    print("Trop de pages vides consécutives, arrêt.")
                    break
            else:
                consecutive_empty = 0

            for product in products:
                if is_valid_product(product):
                    info = extract_product_info(product)
                    valid_products.append(info)

                    image_url = info[-1]
                    image_id = info[0]

                    task = asyncio.create_task(
                        download_image(session, image_url, image_id, sem_img, folder)
                    )
                    image_tasks.append(task)

                    if len(valid_products) >= target_count:
                        break

            page += 1

        await asyncio.gather(*image_tasks)
        return valid_products


# -------------------------
# CSV export
# -------------------------
def save_to_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["foodId", "label", "category", "foodContentsLabel", "image"])
        writer.writerows(rows)


# -------------------------
# Entry point
# -------------------------
def main():
    products = asyncio.run(scrape(CATEGORY, TARGET_COUNT, PAGE_SIZE, MAX_PAGES))
    output_file = f"data/raw/metadata_{CATEGORY}_{TARGET_COUNT}.csv"
    save_to_csv(output_file, products)
    print(f"✔ Fichier {output_file} créé. Produits valides collectés : {len(products)}")


if __name__ == "__main__":
    main()
