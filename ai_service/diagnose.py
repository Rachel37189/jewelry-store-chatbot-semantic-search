import requests, warnings, re
warnings.filterwarnings('ignore')

BASE = 'https://localhost:44320'
SEARCH = 'http://localhost:8001/search'

r = requests.get(f'{BASE}/api/products', verify=False, timeout=10)
products = r.json()

product_list = [
    {
        'id': p.get('productId'),
        'name': p.get('productName', ''),
        'description': p.get('shortDescription', ''),
        'longDescription': p.get('longDescription', ''),
        'color': p.get('color', ''),
        'image1': p.get('image1', ''),
        'inStock': bool(p.get('sizes'))
    }
    for p in products
]

def tokenize(text):
    return set(re.findall(r'[\u05d0-\u05eaa-z0-9]+', text.lower()))

def get_text(p):
    return ' '.join(filter(None,[p['name'],p['description'],p['longDescription'],p['color']]))

# Check which products contain each word as a whole token
for word in ['\u05e4\u05e8\u05e4\u05e8', '\u05e4\u05e8\u05d7', '\u05e2\u05dc\u05d9\u05dd', '\u05e2\u05dc\u05d9\u05d4']:
    matches = [p for p in product_list if word in tokenize(get_text(p))]
    print(f'"{word}" whole-word matches: {len(matches)}')
    for m in matches:
        print(f'  id={m["id"]} name={m["name"]}')
    print()

# Now call search for each and see what comes back
for q in ['\u05e4\u05e8\u05e4\u05e8', '\u05e4\u05e8\u05d7', '\u05e2\u05dc\u05d9\u05d9\u05dd', '\u05e2\u05dc\u05d9\u05d4']:
    print(f'SEARCH "{q}":')
    try:
        resp = requests.post(SEARCH, json={'query': q, 'products': product_list, 'top_k': 50}, timeout=60)
        results = resp.json().get('results', [])
        print(f'  {len(results)} results:')
        for item in results:
            print(f'    score={item.get("score")}  id={item.get("id")}  name={item.get("name")}')
        if not results:
            print('  (none)')
    except Exception as e:
        print(f'  Error: {e}')
    print()
