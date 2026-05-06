#!/usr/bin/env python3
"""Nude Project Scraper - With smart batch processing, stale detection, and embedding optimization"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Set
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import torch
from PIL import Image
import io
import numpy as np
from transformers import AutoModel, AutoProcessor
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4")
SOURCE = "scraper-nudeproject"
BRAND = "Nude Project"

CATEGORY_URLS = [
    ("https://nude-project.com/collections/accessories", None),
    ("https://nude-project.com/collections/t-shirts", None),
    ("https://nude-project.com/collections/shirts-polos", None),
    ("https://nude-project.com/collections/hoodies", None),
    ("https://nude-project.com/collections/knitwear", None),
    ("https://nude-project.com/collections/jeans", "men"),
    ("https://nude-project.com/collections/pants", None),
    ("https://nude-project.com/collections/shorts", "men"),
    ("https://nude-project.com/collections/swimwear", "men"),
    ("https://nude-project.com/collections/outerwear", None),
    ("https://nude-project.com/collections/womens-exclusive-tops-t-shirts", "women"),
    ("https://nude-project.com/collections/womens-exclusive-bags-leather-goods", "women"),
    ("https://nude-project.com/collections/womens-knitwear-sweatshirts", "women"),
    ("https://nude-project.com/collections/womens-swimwear", "women"),
    ("https://nude-project.com/collections/womens-exclusive-bottoms", "women"),
    ("https://nude-project.com/collections/womens-jewelry", "women"),
    ("https://nude-project.com/collections/womens-exclusive-accessories", "women"),
    ("https://nude-project.com/collections/womens-underwear", "women"),
    ("https://nude-project.com/collections/womens-exclusive-outerwear", "women"),
]

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
SESSION.mount('http://', HTTPAdapter(max_retries=retries))
SESSION.mount('https://', HTTPAdapter(max_retries=retries))
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
})

BATCH_SIZE = 50
EMBED_DELAY = 0.5
MAX_RETRIES = 3

logging.basicConfig(filename="scraper.log", level=logging.INFO)


class SigLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP on {self.device}")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-384", trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        self.model.eval()
    
    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        return emb / (torch.sqrt((emb ** 2).sum(-1, keepdim=True)) + 1e-8)
    
    def embed_image(self, url: str) -> Optional[list]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = SESSION.get(url, timeout=20)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                inp = self.processor(images=img, return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                with torch.no_grad():
                    out = self.model.get_image_features(**inp)
                    emb = self._normalize(out.pooler_output)
                return emb.squeeze().cpu().numpy().tolist()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Img error: {e}")
                    return None
                time.sleep(1)
                continue
    
    def embed_text(self, text: str) -> Optional[list]:
        try:
            text = text[:500]
            inp = self.processor(text=text, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model.get_text_features(**inp)
                emb = self._normalize(out.pooler_output)
            return emb.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Txt error: {e}")
            return None


def get_urls(category_url: str) -> list:
    urls = []
    url_set = set()
    page = 1
    max_retries = 3
    
    while page <= 50:
        url = category_url if page == 1 else f"{category_url}?page={page}"
        
        for attempt in range(max_retries):
            try:
                r = SESSION.get(url, timeout=20)
                if r.status_code != 200:
                    break
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  Failed after {max_retries} attempts, skipping page {page}")
                    return urls
                time.sleep(2)
                continue
        
        if r.status_code != 200:
            break
            
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select('a[href*="/products/"]')
        
        new_urls = []
        for l in links:
            h = l.get("href", "")
            if "/products/" in h:
                full = urljoin("https://nude-project.com", h)
                if full.startswith("https://nude-project.com") and full not in url_set:
                    url_set.add(full)
                    new_urls.append(full)
        
        if not new_urls:
            break
            
        urls.extend(new_urls)
        print(f"  Page {page}: {len(new_urls)} new URLs (total: {len(urls)})")
        page += 1
        time.sleep(0.2)
    
    return urls


def extract(url: str) -> Optional[dict]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code != 200:
                return None
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Failed to fetch after {max_retries} attempts: {url}")
                return None
            time.sleep(2)
            continue
    
    matches = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.+?)</script>', r.text, re.DOTALL)
    pd = {}
    for m in matches:
        try:
            d = json.loads(m)
            if d.get('@type') == 'Product':
                pd = d
                break
        except:
            pass
    
    if not pd:
        return None
    
    title = pd.get('name', '')
    desc = pd.get('description', '')
    
    brand = pd.get('brand', {})
    vendor = brand.get('name') if isinstance(brand, dict) else BRAND
    
    images = pd.get('image')
    main_image = None
    if isinstance(images, dict):
        main_image = images.get('url')
    elif isinstance(images, list) and images:
        main_image = images[0].get('url') if isinstance(images[0], dict) else images[0]
    elif isinstance(images, str):
        main_image = images
    
    if main_image and not main_image.startswith('http'):
        main_image = urljoin("https://nude-project.com", main_image)
    
    offers = pd.get('offers', [])
    prices = list(dict.fromkeys([o.get('price', '0') for o in offers]))
    sizes = list(dict.fromkeys([o.get('name', '') for o in offers if o.get('name')]))
    
    if prices:
        price_val = prices[0].replace(',', '.').strip()
        try:
            price_float = float(price_val)
            formatted_price = f"{price_float:.2f}EUR"
        except:
            formatted_price = prices[0]
    else:
        formatted_price = '0'
    
    category = pd.get('category', '')
    if not category:
        try:
            category = url.split('/collections/')[1].split('/products/')[0].replace('-', ' ').title()
        except:
            category = ''
    
    tags = pd.get('keywords', [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',')]
    
    gender = None
    if 'women' in url.lower() or 'womens' in url.lower():
        gender = 'women'
    elif 'men' in url.lower() or 'mens' in url.lower():
        gender = 'men'
    
    handle = url.split('/products/')[-1]
    
    metadata = json.dumps({
        'title': title, 'description': desc, 'vendor': vendor,
        'category': category, 'prices': prices, 'sizes': sizes, 'tags': tags,
    })
    
    return {
        'id': f'nudeproject-{handle}',
        'source': SOURCE,
        'product_url': url,
        'image_url': main_image,
        'brand': vendor,
        'title': title,
        'description': desc,
        'category': category,
        'gender': gender,
        'metadata': metadata,
        'size': ', '.join(sizes),
        'price': formatted_price,
        'tags': tags,
    }


def get_existing_products(supabase) -> dict:
    """Fetch all existing products for this source"""
    existing = {}
    try:
        result = supabase.table('products').select('id, product_url, title, image_url, price, created_at').eq('source', SOURCE).execute()
        for p in result.data:
            existing[p['product_url']] = p
    except Exception as e:
        print(f"Error fetching existing: {e}")
    return existing


def batch_upsert(supabase, products: list) -> dict:
    """Insert/update products in batches with retry logic"""
    results = {'success': 0, 'failed': 0, 'failed_ids': []}
    timestamp = datetime.utcnow().isoformat()
    
    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i + BATCH_SIZE]
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                data = []
                for p in batch:
                    record = {
                        'id': p['id'],
                        'source': p['source'],
                        'product_url': p['product_url'],
                        'image_url': p['image_url'],
                        'brand': p['brand'],
                        'title': p['title'],
                        'description': p['description'],
                        'category': p['category'],
                        'gender': p['gender'],
                        'metadata': p['metadata'],
                        'size': p['size'],
                        'second_hand': False,
                        'image_embedding': p.get('image_embedding'),
                        'country': None,
                        'tags': p['tags'],
                        'price': p['price'],
                        'info_embedding': p.get('info_embedding'),
                    }
                    data.append(record)
                
                supabase.table('products').upsert(data, on_conflict='id').execute()
                results['success'] += len(batch)
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"Batch failed after {MAX_RETRIES} retries: {e}")
                    for p in batch:
                        results['failed'] += 1
                        results['failed_ids'].append(p['id'])
                        logging.error(f"Failed product: {p['id']} - {e}")
                else:
                    time.sleep(1)
    
    return results


def delete_stale_products(supabase, seen_urls: Set[str]) -> int:
    """Delete products not seen in current run"""
    deleted = 0
    try:
        result = supabase.table('products').select('id, product_url, created_at').eq('source', SOURCE).execute()
        
        for p in result.data:
            if p['product_url'] not in seen_urls:
                supabase.table('products').delete().eq('id', p['id']).execute()
                deleted += 1
                print(f"Deleted stale: {p['id']}")
        
    except Exception as e:
        print(f"Error deleting stale: {e}")
    
    return deleted


def check_changed(existing_product: dict, new_product: dict) -> bool:
    """Check if product has actually changed"""
    if not existing_product:
        return True
    
    if existing_product.get('title') != new_product.get('title'):
        return True
    if existing_product.get('image_url') != new_product.get('image_url'):
        return True
    if existing_product.get('price') != new_product.get('price'):
        return True
    
    return False


def main():
    embedder = SigLIPEmbedder()
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("\n=== Fetching existing products ===")
    existing = get_existing_products(supabase)
    print(f"Found {len(existing)} existing products")
    
    all_products = []
    seen_urls = set()
    
    for cat_url, gender in CATEGORY_URLS:
        cat_name = cat_url.split('/collections/')[-1]
        print(f"\n=== Processing: {cat_name} ===")
        
        urls = get_urls(cat_url)
        print(f"Found {len(urls)} products")
        
        for url in urls:
            p = extract(url)
            if p:
                p['gender'] = p['gender'] or gender
                all_products.append(p)
                seen_urls.add(url)
    
    print(f"\nTotal scraped: {len(all_products)}")
    
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    
    products_to_insert = []
    
    print("\n=== Processing products ===")
    for i, p in enumerate(all_products):
        print(f"  {i+1}/{len(all_products)}: {p['title']}")
        
        existing_p = existing.get(p['product_url'])
        
        if not existing_p:
            print(f"    NEW product")
            new_count += 1
            generate_embeddings = True
        elif check_changed(existing_p, p):
            print(f"    CHANGED - will update")
            updated_count += 1
            generate_embeddings = True
        else:
            print(f"    UNCHANGED - skipping")
            unchanged_count += 1
            generate_embeddings = False
            p['image_embedding'] = None
            p['info_embedding'] = None
        
        if generate_embeddings:
            if p.get('image_url'):
                p['image_embedding'] = embedder.embed_image(p['image_url'])
                time.sleep(EMBED_DELAY)
            
            txt = f"{p.get('title', '')} {p.get('description', '')} {p.get('category', '')} {p.get('gender', '')} {p.get('price', '')}"
            p['info_embedding'] = embedder.embed_text(txt)
            time.sleep(EMBED_DELAY)
        
        products_to_insert.append(p)
        
        if len(products_to_insert) >= BATCH_SIZE:
            print(f"\n  Inserting batch of {len(products_to_insert)}...")
            unique_products = {p['id']: p for p in products_to_insert}.values()
            result = batch_upsert(supabase, list(unique_products))
            products_to_insert = []
    
    if products_to_insert:
        print(f"\n  Inserting final batch of {len(products_to_insert)}...")
        unique_products = {p['id']: p for p in products_to_insert}.values()
        result = batch_upsert(supabase, list(unique_products))
    
    print("\n=== Deleting stale products ===")
    stale_deleted = delete_stale_products(supabase, seen_urls)
    
    print("\n" + "="*50)
    print("RUN SUMMARY")
    print("="*50)
    print(f"New products added:     {new_count}")
    print(f"Products updated:   {updated_count}")
    print(f"Unchanged:       {unchanged_count}")
    print(f"Stale deleted:   {stale_deleted}")
    print("="*50)


if __name__ == "__main__":
    main()