#!/usr/bin/env python3
"""Nude Project Scraper - Complete solution for scraping, embedding, and database insertion"""

import os
import re
import json
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})
SESSION.params.update({"noajax": "1"})


class SigLIPEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-384"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP model: {model_name} on {self.device}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(emb ** 2, dim=-1, keepdim=True))
        return emb / (norm + 1e-8)
    
    def embed_image(self, image_url: str) -> Optional[np.ndarray]:
        try:
            response = SESSION.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
                emb = out.pooler_output
                emb = self._normalize(emb)
            
            return emb.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error embedding image {image_url}: {e}")
            return None
    
    def embed_images(self, image_urls: list) -> Optional[np.ndarray]:
        embeddings = []
        for url in image_urls:
            emb = self.embed_image(url)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        try:
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                emb = out.pooler_output
                emb = self._normalize(emb)
            
            return emb.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None


class NudeProjectScraper:
    def __init__(self):
        self.base_url = "https://nude-project.com"
    
    def get_category_products(self, category_url: str) -> list:
        product_urls = []
        page = 1
        
        while True:
            if page == 1:
                url = category_url
            else:
                url = f"{category_url}?page={page}"
            
            print(f"Scraping category: {url}")
            
            try:
                response = SESSION.get(url, timeout=30)
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.text, "html.parser", from_encoding="utf-8")
                
                links = soup.select('a[href*="/products/"]')
                found_any = False
                
                for link in links:
                    href = link.get("href", "")
                    if "/products/" in href and href not in product_urls:
                        full_url = urljoin(self.base_url, href)
                        if full_url.startswith(self.base_url):
                            product_urls.append(full_url)
                            found_any = True
                
                if not found_any:
                    break
                
                page += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                break
        
        return list(set(product_urls))
    
    def get_product(self, product_url: str, gender_hint: str = None) -> Optional[dict]:
        print(f"Scraping product: {product_url}")
        
        response = SESSION.get(product_url, timeout=30)
        if response.status_code != 200:
            return None
        
        return self._extract_from_html(response.text, product_url, gender_hint)
    
    def _extract_from_html(self, html: str, url: str, gender_hint: str = None) -> Optional[dict]:
        base_url = self.base_url
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
        
        gender = gender_hint
        if not gender:
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


class DatabaseManager:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def insert_product(self, product: dict) -> bool:
        try:
            data = {
                'id': product['id'],
                'source': product['source'],
                'product_url': product['product_url'],
                'affiliate_url': product['affiliate_url'],
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
                'compressed_image_url': product['compressed_image_url'],
                'tags': product['tags'],
                'price': product['price'],
                'sale': product['sale'],
                'additional_images': product['additional_images'],
                'info_embedding': product['info_embedding'],
            }
            
            result = self.supabase.table('products').upsert(data, on_conflict='id').execute()
            print(f"Inserted: {product['title']}")
            return True
        except Exception as e:
            print(f"Error inserting {product['title']}: {e}")
            return False
    
    def batch_insert(self, products: list) -> int:
        count = 0
        for product in products:
            if self.insert_product(product):
                count += 1
        return count


def main():
    print("=" * 60)
    print("Nude Project Scraper - Starting")
    print("=" * 60)
    
    embedder = SigLIPEmbedder()
    scraper = NudeProjectScraper()
    db = DatabaseManager()
    
    all_products = []
    
    for category_url, gender_hint in CATEGORY_URLS:
        category_name = category_url.split('/collections/')[-1]
        print(f"\n{'='*60}")
        print(f"Processing: {category_name}")
        print(f"{'='*60}")
        
        product_urls = scraper.get_category_products(category_url)
        print(f"Found {len(product_urls)} products")
        
        for url in product_urls:
            try:
                product = scraper.get_product(url, gender_hint)
                if product:
                    all_products.append(product)
                    print(f"  Extracted: {product['title']}")
                time.sleep(0.3)
            except Exception as e:
                print(f"  Error: {url} - {e}")
        
        time.sleep(1)
    
    print(f"\nTotal products: {len(all_products)}")
    
    print("\nGenerating embeddings...")
    for i, product in enumerate(all_products):
        print(f"Processing {i+1}/{len(all_products)}: {product['title']}")
        
        img_embs = []
        if product.get('image_url'):
            img_emb = embedder.embed_image(product['image_url'])
            if img_emb is not None:
                img_embs.append(img_emb)
        
        if product.get('additional_images'):
            add_imgs = [img.strip() for img in product['additional_images'].split(',')]
            for img_url in add_imgs[:5]:
                if img_url and img_url != product.get('image_url'):
                    emb = embedder.embed_image(img_url)
                    if emb is not None:
                        img_embs.append(emb)
        
        if img_embs:
            combined = np.mean(img_embs, axis=0)
            product['image_embedding'] = combined.tolist()
            print(f"  Image embedding: {len(product['image_embedding'])} dims")
        
        info_text = f"{product.get('title', '')} {product.get('description', '')} {product.get('category', '')} {product.get('gender', '')} {product.get('price', '')}"
        text_emb = embedder.embed_text(info_text)
        if text_emb is not None:
            product['info_embedding'] = text_emb.tolist()
            print(f"  Text embedding: {len(product['info_embedding'])} dims")
        
        time.sleep(0.2)
    
    print("\nInserting to database...")
    count = db.batch_insert(all_products)
    print(f"\nCompleted! Inserted {count} products")


if __name__ == "__main__":
    main()