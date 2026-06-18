from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re
import hashlib

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


SYSTEM_PROMPT = f"""
You are Gemma, a personal jewelry stylist at {os.getenv('STORE_NAME')}.
{os.getenv('STORE_DESCRIPTION')}
Your tone is warm, elegant, and knowledgeable - like a trusted friend with an eye for beauty, not a pushy salesperson.

Rules you MUST always follow:
- Never recommend a product we don't carry. Only mention rings, necklaces, bracelets, and earrings from our handcrafted collection.
- Always ask about the customer's budget before making recommendations.
- Keep answers to 3-4 sentences maximum.
- All prices are in Israeli shekels (ג‚×). Never use $ or the word "dollars".
- When mentioning a product by name, always use its markdown link exactly as shown in the product list, for example: [׳¦׳׳™׳“ ׳™׳“ ׳•׳¨׳•׳“](/product/12).
- If the customer mentions a competitor or another store, say: "I only know our own collection, but I'd love to help you find something just as beautiful here!"
- Always end every reply with exactly one follow-up question.
- If you don't know something, say so honestly.

When comparing two products, use this exact structure:
Option A: [name] - [one sentence benefit]
Option B: [name] - [one sentence benefit]
My pick: [which one and why, one sentence]

Example conversation:
User: I'm looking for a gift for my mom's birthday.
Gemma: What a thoughtful gift! Our handcrafted pieces make wonderful birthday surprises. What is your budget so I can point you to our best options for her?
User: Around $100.
Gemma: Perfect, $100 opens up some beautiful choices! Our Delicate Pearl Necklace ($89) is a timeless classic, and our Gold Hoop Earrings ($95) are effortlessly elegant for everyday wear. Does your mom prefer classic and understated, or something a little bolder?
"""


# -------------------------
# CACHES
# -------------------------

_embedding_cache: dict[str, list[float]] = {}
_product_embedding_cache: dict[str, list[float]] = {}
_query_expansion_cache: dict[str, str] = {}


def _product_cache_key(pid: str, text: str) -> str:
    """
    Cache key includes a hash of the product text.
    This prevents stale embeddings when product text changes.
    """
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
    return f'{pid}:{h}'


def get_embedding(text: str) -> list[float]:
    text = text.strip()

    if not text:
        return []

    if text in _embedding_cache:
        return _embedding_cache[text]

    print(f'[embedding] calling OpenAI for: {text[:80]}')

    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )

    embedding = response.data[0].embedding
    _embedding_cache[text] = embedding
    return embedding


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b:
        return 0.0

    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        return 0.0

    return float(np.dot(a, b) / norm)


# -------------------------
# MODELS
# -------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    products: list = []


class SearchRequest(BaseModel):
    query: str
    products: list = []
    top_k: int = 50


# -------------------------
# TEXT HELPERS
# -------------------------

def get_product_text(p: dict) -> str:
    """
    Collect all possible product text fields.
    This makes the Python service safe even if .NET sends different field names.
    """
    return ' '.join(filter(None, [
        str(p.get('name', '')),
        str(p.get('productName', '')),
        str(p.get('description', '')),
        str(p.get('shortDescription', '')),
        str(p.get('longDescription', '')),
        str(p.get('productDescription', '')),
        str(p.get('color', '')),
        str(p.get('category', '')),
        str(p.get('categoryName', '')),
    ])).strip()


def tokenize(text: str) -> set[str]:
    """
    Split Hebrew / English / numbers into word tokens.
    """
    return set(re.findall(r'[\u05d0-\u05eaa-z0-9]+', text.lower()))


def word_match(term: str, product_tokens: set[str]) -> bool:
    """
    Hebrew-friendly matching.

    Exact match:
      פרפר == פרפר

    Prefix match:
      פרח -> פרחים / פרחוני
      פרפר -> פרפרים
      דג -> דגים
      עגיל -> עגילים / עגילי

    Not substring match:
      דג does NOT match גדול
      פר does NOT match פרחוני unless the term itself is long enough
    """
    term = term.lower().strip()

    if not term:
        return False

    for token in product_tokens:
        if token == term:
            return True

        # Prefix match, not substring match.
        # len >= 3 prevents tiny terms from matching too broadly.
        if len(term) >= 3 and token.startswith(term):
            return True

        # Also allow token -> term prefix for cases like:
        # query: עגילים, product token: עגילי
        if len(token) >= 3 and term.startswith(token):
            return True

    return False


def keyword_boost(query_words: list[str], product_tokens: set[str]) -> float:
    """
    0.0 - 0.3 boost for keyword matches.
    """
    words = [w.strip() for w in query_words if len(w.strip()) > 1]

    if not words:
        return 0.0

    hits = sum(1 for w in words if word_match(w, product_tokens))
    return (hits / len(words)) * 0.3


# -------------------------
# QUERY EXPANSION
# -------------------------

def expand_query(query: str) -> tuple[str, list[str]]:
    """
    Expands broad category queries.
    Specific item queries stay unchanged.

    Returns:
      expanded_text, expanded_terms

    If expanded_terms is empty -> specific query.
    If expanded_terms has values -> broad semantic query.
    """
    query = query.strip()

    if query in _query_expansion_cache:
        cached = _query_expansion_cache[query]
        terms = cached.split() if cached != query else []
        return cached, terms

    system_msg = (
        "You are a semantic search helper for a jewelry store. "
        "The user typed a search query. It may be in any language.\n\n"

        "Decide: BROAD CATEGORY or SPECIFIC ITEM?\n\n"

        "BROAD CATEGORY = a general group or meaning, for example: "
        "animals, musical instruments, nature, flowers/plants, sports, seasons, holidays, love, emotions, gifts.\n"
        "For a BROAD CATEGORY: list 8-10 specific members of that category in the exact same language as the query. "
        "Include the original query words. "
        "Focus on the design theme: specific things that could appear on a product. "
        "Never add general jewelry types such as ring, earring, bracelet, necklace, unless the query itself is about jewelry types.\n\n"

        "SPECIFIC ITEM = already a concrete thing or direct product search, for example: "
        "butterfly, guitar, flower, heart, gold ring, silver earrings.\n"
        "For a SPECIFIC ITEM: return the query exactly unchanged.\n\n"

        "Return only words, space-separated. No explanation. No punctuation."
    )

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': f'Query: {query}'}
        ],
        max_tokens=80,
        temperature=0,
    )

    expanded = response.choices[0].message.content.strip()
    _query_expansion_cache[query] = expanded

    terms = expanded.split() if expanded != query else []

    print(f'[search] expand: "{query}" -> "{expanded}"')
    return expanded, terms


# -------------------------
# CHAT ENDPOINT
# -------------------------

@app.post('/chat')
async def chat(req: ChatRequest):
    if req.products:
        catalog_lines = []

        for p in req.products:
            stock = 'in stock' if p.get('inStock') else 'out of stock'
            pid = p.get('id') or p.get('productId') or ''
            name = p.get('name') or p.get('productName') or ''
            price = p.get('price') or p.get('productPrice') or ''
            color = p.get('color', '')
            description = p.get('description') or p.get('shortDescription') or ''

            line = f"- [{name}](/product/{pid}) (₪{price}) [{stock}] | Color: {color} | {description}"
            catalog_lines.append(line)

        catalog = '\n'.join(catalog_lines)
        full_prompt = SYSTEM_PROMPT + f"\n\nAvailable products:\n{catalog}\nOnly recommend products from this list."
    else:
        full_prompt = SYSTEM_PROMPT

    messages = [{'role': 'system', 'content': full_prompt}]

    for m in req.history:
        messages.append({'role': m.role, 'content': m.content})

    messages.append({'role': 'user', 'content': req.message})

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        max_tokens=400,
        temperature=0.8,
    )

    return {'reply': response.choices[0].message.content}


# -------------------------
# SEARCH ENDPOINT
# -------------------------

@app.post('/search')
async def search(req: SearchRequest):
    if not req.products:
        return {'results': []}

    query = req.query.strip()

    if not query:
        return {'results': []}

    print(f'\n[search] query="{query}" products={len(req.products)}')

    expanded, expanded_terms = expand_query(query)
    is_broad = len(expanded_terms) > 0

    query_embedding = get_embedding(expanded)

    # For specific queries use original query words.
    # For broad queries use expanded terms.
    query_words = query.split()
    boost_words = expanded_terms if is_broad else query_words

    scored = []

    for p in req.products:
        product_text = get_product_text(p)
        tokens = tokenize(product_text)

        pid = str(
            p.get('id')
            or p.get('productId')
            or p.get('productID')
            or p.get('product_id')
            or p.get('name')
            or p.get('productName')
            or ''
        )

        cache_key = _product_cache_key(pid, product_text)

        if cache_key not in _product_embedding_cache:
            print(f'[search] embedding product id={pid} name={(p.get("name") or p.get("productName") or "")[:40]}')
            _product_embedding_cache[cache_key] = get_embedding(product_text)

        product_embedding = _product_embedding_cache[cache_key]
        semantic = cosine_similarity(query_embedding, product_embedding)

        boost = keyword_boost(boost_words, tokens)
        has_keyword_match = boost > 0

        combined = round(min(semantic + boost, 1.0), 3)

        scored.append({
            **p,
            'score': combined,
            '_sem': semantic,
            '_kw': has_keyword_match,
            '_text': product_text,
        })

    scored.sort(key=lambda x: x['score'], reverse=True)

    kw_matched = [r for r in scored if r['_kw']]
    no_match = [r for r in scored if not r['_kw']]

    print(f'[search] keyword matches: {len(kw_matched)}, no match: {len(no_match)}, broad={is_broad}')

    for r in kw_matched:
        print(
            f'  kw_match: score={r["score"]} '
            f'sem={round(r["_sem"], 3)} '
            f'name={r.get("name") or r.get("productName") or ""}'
        )

    # if kw_matched:
    #     # Important:
    #     # Every keyword-matched product is returned.
    #     # Semantic score is only used for ordering, not filtering.
    #     results = kw_matched[:req.top_k]
    # else:
    #     if is_broad:
    #         # For broad queries, do NOT fallback to pure semantic.
    #         # This prevents unrelated general jewelry from coming back.
    #         print('[search] broad query with no keyword matches -> returning no results')
    #         results = []
    #     else:
    #         # For direct/specific queries, allow strict semantic fallback.
    #         print('[search] no keyword matches, using strict semantic fallback')
    #         results = [r for r in scored if r['_sem'] >= 0.42][:req.top_k]
    if is_broad:
        # Broad query: allow keyword matches,
        # or only very strong semantic matches.
        results = [
            r for r in scored
            if r['_kw'] or r['_sem'] >= 0.55
        ][:req.top_k]
    else:
        # Specific query: allow keyword matches,
        # or regular semantic matches.
        results = [
            r for r in scored
            if r['_kw'] or r['_sem'] >= 0.42
        ][:req.top_k]
        for r in results:
            r.pop('_sem', None)
            r.pop('_kw', None)
            r.pop('_text', None)

    print(f'[search] returning {len(results)} results for "{query}"')
    return {'results': results}