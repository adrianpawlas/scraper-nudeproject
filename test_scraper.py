#!/usr/bin/env python3
"""Nude Project Scraper - Test Mode with sample products"""

import re
import json
import time
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

SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"
SOURCE = "scraper-nudeproject"
BRAND = "Nude Project"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


class SigLIPEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-384"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(emb ** 2, dim=-1, keepdim=True))
        return emb / (norm + 1e-8)
    
    def embed_image(self, image_url: str) -> Optional[list]:
        try:
            response = SESSION.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
                emb = self._normalize(out.pooler_output)
            return emb.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Image embed error: {e}")
            return None
    
    def embed_text(self, text: str) -> Optional[list]:
        try:
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                emb = self._normalize(out.pooler_output)
            return emb.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Text embed error: {e}")
            return None


def extract_product(url: str) -> Optional[dict]:
    print(f"Fetching: {url}")
    
    response = SESSION.get(url, timeout=30)
    if response.status_code != 200:
        print(f"Failed: {response.status_code}")
        return None
    
    return _extract_from_html(response.text, url)


def _extract_from_html(html: str, url: str) -> Optional[dict]:
    base_url = "https://nude-project.com"
    soup = BeautifulSoup(html, "html.parser")
    
    matches = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.+?)</script>', html, re.DOTALL)
    
    product_data = {}
    for m in matches:
        try:
            data = json.loads(m)
            if data.get('@type') == 'Product':
                product_data = data
                break
        except:
            pass
    
    if not product_data:
        return None
    
    title = product_data.get('name', '')
    desc = product_data.get('description', '')
    
    brand = product_data.get('brand', {})
    if isinstance(brand, dict):
        brand = brand.get('name')
    vendor = brand or BRAND
    
    images = product_data.get('image')
    if isinstance(images, dict):
        main_image = images.get('url')
    elif isinstance(images, list) and images:
        if isinstance(images[0], dict):
            main_image = images[0].get('url')
        else:
            main_image = images[0]
    elif isinstance(images, str):
        main_image = images
    else:
        main_image = None
    
    if main_image and not main_image.startswith('http'):
        main_image = urljoin(base_url, main_image)
    
    additional_images = []
    if isinstance(images, list) and len(images) > 1:
        for img in images[1:]:
            if isinstance(img, dict):
                img_url = img.get('url')
            else:
                img_url = str(img)
            if img_url:
                if not img_url.startswith('http'):
                    img_url = urljoin(base_url, img_url)
                additional_images.append(img_url)
    elif images:
        data_imgs = soup.find_all('img', {'data-src': True})
        for img in data_imgs[:10]:
            src = img.get('data-src', '')
            if src and 'product' in src.lower() and src != main_image:
                if src not in additional_images:
                    additional_images.append(src)
    
    additional_images_str = ", ".join(additional_images) if additional_images else None
    
    offers = product_data.get('offers', [])
    prices = []
    sizes = []
    for offer in offers:
        prices.append(offer.get('price', '0'))
        var_name = offer.get('name', '')
        if var_name:
            sizes.append(var_name)
    
    unique_prices = list(dict.fromkeys(prices))
    unique_sizes = list(dict.fromkeys(sizes))
    
    original_price = unique_prices[0] if unique_prices else "0"
    
    category = product_data.get('category', '')
    if not category:
        url_path = url.split('/collections/')[-1].split('/products/')[0] if '/collections/' in url else ''
        category = url_path.replace('-', ' ').title()
    
    tags = product_data.get('keywords', [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(',')]
    
    gender = None
    url_lower = url.lower()
    if 'women' in url_lower or 'womens' in url_lower:
        gender = 'women'
    elif 'men' in url_lower or 'mens' in url_lower:
        gender = 'men'
    
    handle = url.split('/products/')[-1] if '/products/' in url else ''
    
    metadata = json.dumps({
        'title': title,
        'description': desc,
        'vendor': vendor,
        'category': category,
        'prices': unique_prices,
        'size_variants': unique_sizes,
        'tags': tags,
    })
    
    return {
        'id': f'nudeproject-{handle}',
        'source': SOURCE,
        'product_url': url,
        'affiliate_url': None,
        'image_url': main_image,
        'brand': vendor,
        'title': title,
        'description': desc,
        'category': category,
        'gender': gender,
        'metadata': metadata,
        'size': ', '.join(unique_sizes),
        'second_hand': False,
        'image_embedding': None,
        'country': 'ES',
        'compressed_image_url': None,
        'tags': tags,
        'price': original_price,
        'sale': None,
        'additional_images': additional_images_str,
        'info_embedding': None,
    }


def insert_product(product: dict) -> bool:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        data = {
            'id': product['id'],
            'source': product['source'],
            'product_url': product['product_url'],
            'image_url': product['image_url'],
            'brand': product['brand'],
            'title': product['title'],
            'description': product['description'],
            'category': product['category'],
            'gender': product['gender'],
            'metadata': product['metadata'],
            'size': product['size'],
            'second_hand': product['second_hand'],
            'image_embedding': product['image_embedding'],
            'country': product['country'],
            'tags': product['tags'],
            'price': product['price'],
            'sale': product['sale'],
            'additional_images': product['additional_images'],
            'info_embedding': product['info_embedding'],
        }
        
        result = supabase.table('products').upsert(data, on_conflict='id').execute()
        print(f"Inserted: {product['title']}")
        return True
    except Exception as e:
        print(f"Error inserting: {e}")
        return False


if __name__ == "__main__":
    SAMPLE_URLS = [
        "https://nude-project.com/collections/accessories/products/venecia-cap",
        "https://nude-project.com/collections/t-shirts/products/bear-ink-tee-marshmallow",
        "https://nude-project.com/collections/shirts-polos/products/amalfi-shirt",
    ]
    
    print("=" * 50)
    print("Testing Nude Project Scraper")
    print("=" * 50)
    
    embedder = SigLIPEmbedder()
    
    for url in SAMPLE_URLS:
        print(f"\n--- Processing: {url} ---")
        
        product = extract_product(url)
        if not product:
            print("Failed to extract product")
            continue
        
        print(f"Title: {product['title']}")
        print(f"Price: {product['price']}")
        print(f"Image: {product['image_url']}")
        print(f"Additional Images: {product['additional_images']}")
        
        img_embs = []
        if product['image_url']:
            print("Generating image embedding for main image...")
            img_emb = embedder.embed_image(product['image_url'])
            if img_emb:
                img_embs.append(img_emb)
                print(f"  Main image embedding: {len(img_emb)} dims OK")
        
        if product.get('additional_images'):
            add_imgs = [img.strip() for img in product['additional_images'].split(',')]
            print(f"Generating embeddings for {len(add_imgs)} additional images...")
            for i, img_url in enumerate(add_imgs[:5]):
                if img_url and img_url != product.get('image_url'):
                    emb = embedder.embed_image(img_url)
                    if emb:
                        img_embs.append(emb)
                        print(f"  Additional image {i+1} embedding OK")
        
        if img_embs:
            combined_emb = list(np.mean(img_embs, axis=0))
            product['image_embedding'] = combined_emb
            print(f"  Combined image embedding: {len(combined_emb)} dims")
        
        info_text = f"{product['title']} {product['description']} {product['category']} {product['gender']} {product['price']}"
        print("Generating text embedding...")
        text_emb = embedder.embed_text(info_text)
        if text_emb:
            product['info_embedding'] = text_emb
            print(f"  Text embedding: {len(text_emb)} dims OK")
        
        print(f"\nInserting to Supabase...")
        insert_product(product)
        
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)