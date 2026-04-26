#!/usr/bin/env python3
"""Nude Project Scraper - Optimized with batch processing and progress saving"""

import os
import re
import json
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import torch
from PIL import Image
import io
import numpy as np
from transformers import AutoModel, AutoProcessor
from supabase import create_client
import pickle

SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"
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

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
})


class SigLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP on {self.device}")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-384", trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        self.model.eval()
    
    def _normalize(self, emb):
        return emb / (torch.sqrt((emb ** 2).sum(-1, keepdim=True)) + 1e-8)
    
    def embed_image(self, url):
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
            print(f"Img error: {e}")
            return None
    
    def embed_text(self, text):
        try:
            inp = self.processor(text=text, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model.get_text_features(**inp)
                emb = self._normalize(out.pooler_output)
            return emb.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Txt error: {e}")
            return None


def get_urls(category_url):
    urls = []
    page = 1
    while page <= 50:
        url = category_url if page == 1 else f"{category_url}?page={page}"
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select('a[href*="/products/"]')
        
        new_urls = set()
        for l in links:
            h = l.get("href", "")
            if "/products/" in h and h not in urls:
                full = urljoin("https://nude-project.com", h)
                if full.startswith("https://nude-project.com"):
                    new_urls.add(full)
        
        if not new_urls:
            break
            
        urls.extend(list(new_urls))
        print(f"  Page {page}: {len(new_urls)} new URLs (total: {len(urls)})")
        page += 1
        time.sleep(0.2)
    
    return urls


def extract(url):
    r = SESSION.get(url, timeout=20)
    if r.status_code != 200:
        return None
    
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
        'source': SOURCE, 'product_url': url, 'affiliate_url': None,
        'image_url': main_image, 'brand': vendor, 'title': title,
        'description': desc, 'category': category, 'gender': gender,
        'metadata': metadata, 'size': ', '.join(sizes),
        'second_hand': False, 'image_embedding': None,
        'country': 'ES', 'compressed_image_url': None, 'tags': tags,
        'price': prices[0] if prices else '0', 'sale': None,
        'additional_images': None, 'info_embedding': None,
    }


def insert(products):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    count = 0
    for p in products:
        try:
            data = {
                'id': p['id'], 'source': p['source'], 'product_url': p['product_url'],
                'image_url': p['image_url'], 'brand': p['brand'], 'title': p['title'],
                'description': p['description'], 'category': p['category'], 'gender': p['gender'],
                'metadata': p['metadata'], 'size': p['size'], 'second_hand': p['second_hand'],
                'image_embedding': p['image_embedding'], 'country': p['country'],
                'tags': p['tags'], 'price': p['price'], 'sale': p['sale'],
                'additional_images': p['additional_images'], 'info_embedding': p['info_embedding'],
            }
            supabase.table('products').upsert(data, on_conflict='id').execute()
            count += 1
        except Exception as e:
            print(f"Insert error: {e}")
    return count


def main():
    embedder = SigLIPEmbedder()
    
    total_products = []
    
    for cat_url, gender in CATEGORY_URLS:
        cat_name = cat_url.split('/collections/')[-1]
        print(f"\n{'='*50}")
        print(f"Processing: {cat_name}")
        print(f"{'='*50}")
        
        urls = get_urls(cat_url)
        print(f"Found {len(urls)} products")
        
        for i, url in enumerate(urls):
            print(f"  {i+1}/{len(urls)}: {url}")
            p = extract(url)
            if p:
                p['gender'] = p['gender'] or gender
                total_products.append(p)
            time.sleep(0.1)
    
    print(f"\nTotal extracted: {len(total_products)}")
    
    print("\nGenerating embeddings...")
    for i, p in enumerate(total_products):
        print(f"  Embedding {i+1}/{len(total_products)}: {p['title']}")
        
        if p.get('image_url'):
            p['image_embedding'] = embedder.embed_image(p['image_url'])
        
        txt = f"{p.get('title', '')} {p.get('description', '')} {p.get('category', '')} {p.get('gender', '')} {p.get('price', '')}"
        p['info_embedding'] = embedder.embed_text(txt)
        time.sleep(0.1)
    
    print("\nInserting to database...")
    count = insert(total_products)
    print(f"Completed! Inserted {count} products")


if __name__ == "__main__":
    main()