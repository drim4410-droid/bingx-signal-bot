import os
import asyncio
import time
import math
import requests
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var not set")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()

# ---------------------------
# BingX API (Public)
# ---------------------------
BINGX_BASE = "https://open-api.bingx.com"
HTTP_TIMEOUT = 12

# throttling (чтобы не словить лимиты)
MAX_CONCURRENT_REQUESTS = 12
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def _http_get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def bingx_get_usdt_perpetuals() -> list[str]:
    url = f"{BINGX_BASE}/openApi/swap/v2/quote/contracts"
    js = _http_get(url)
    data = js.get("data") or []
    symbols = []
    for item in data:
        quote = item.get("quoteAsset") or item.get("quote") or item.get("quoteCurrency")
        sym = item.get("symbol")
        if sym and (quote == "USDT" or (isinstance(sym, str) and sym.endswith("USDT"))):
            symbols.append(sym)
    # de-dup
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def bingx_get_klines_sync(symbol: str, interval: str = "1h", limit: int = 200):
    url = f"{BINGX_BASE}/openApi/swap/v2/quote/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    js = _http_get(url, params=params)

    candles = js.get("data") or []
    if not candles:
        raise RuntimeError(f"No candles for {symbol} {interval}")

    highs, lows, closes, volumes = [], [], [], []
    for c in candles:
        if isinstance(c, (list, tuple)) and len(c) >= 6:
            high = float(c[2]); low = float(c[3]); close = float(c[4]); vol = float(c[5])
        elif isinstance(c, dict):
            high = float(c.get("high") or c.get("h"))
            low = float(c.get("low") or c.get("l"))
            close = float(c.get("close") or c.get("c"))
            vol = float(c.get("volume") or c.get("v") or 0.0)
        else:
            continue
        highs.append(high); lows.append(low); closes.append(close); volumes.append(vol)

    if len(closes) < 60:
        raise RuntimeError(f"Too few candles for {symbol} {interval}: {len(closes)}")

    return (
        np.array(highs, dtype=np.float64),
        np.array(lows, dtype=np.float64),
        np.array(closes, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
    )

async def bingx_get_klines(symbol: str, interval: str = "1h", limit: int = 200):
    async with SEM:
        return await asyncio.to_thread(bingx_get_klines_sync, symbol, interval, limit)

# ---------------------------
# Indicators
# ---------------------------
def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr.copy()
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 2:
        return float("nan")
    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 2:
        return float("nan")
    prev_close = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)))
    # Wilder smoothing
    atr_val = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
    return float(atr_val)

def swing_structure(closes: np.ndarray, lookback: int = 20) -> str:
    """
    Очень лёгкая оценка структуры:
    - если последние N свечей делают higher-high/higher-low -> bullish
    - lower-low/lower-high -> bearish
    иначе range
    """
    if len(closes) < lookback + 5:
        return "range"
    segment = closes[-lookback:]
    # наклон линейной регрессии как суррогат структуры
    x = np.arange(len(segment))
    slope = np.polyfit(x, segment, 1)[0]
    if slope > 0:
        return "bull"
    if slope < 0:
        return "bear"
    return "range"

# ---------------------------
# Scoring + Signal
# ---------------------------
def score_setup(closes, highs, lows, volumes):
    price = float(closes[-1])

    e20 = ema(closes, 20)[-1]
    e50 = ema(closes, 50)[-1]
    e200 = ema(closes, 200)[-1] if len(closes) >= 200 else ema(closes, 100)[-1]

    r = rsi(closes, 14)
    a = atr(highs, lows, closes, 14)

    # volume spike (последняя свеча vs среднее)
    v_last = float(volumes[-1]) if len(volumes) else 0.0
    v_mean = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else float(np.mean(volumes)) if len(volumes) else 0.0
    vol_spike = (v_last / v_mean) if (v_mean and v_mean > 0) else 0.0

    # direction bias by EMA stack
    bullish_stack = (price > e20 > e50) or (e20 > e50 and price > e50)
    bearish_stack = (price < e20 < e50) or (e20 < e50 and price < e50)

    # trend strength using EMA200
    htf_bull = price > e200
    htf_bear = price < e200

    struct = swing_structure(closes, 25)

    score_long = 0
    score_short = 0

    # HTF bias
    if htf_bull:
        score_long += 20
    if htf_bear:
        score_short += 20

    # EMA alignment
    if bullish_stack:
        score_long += 18
    if bearish_stack:
        score_short += 18

    # Structure
    if struct == "bull":
        score_long += 15
    elif struct == "bear":
        score_short += 15
    else:
        score_long -= 5
        score_short -= 5

    # RSI
    if not math.isnan(r):
        if r > 55:
            score_long += 10
        elif r < 45:
            score_short += 10

    # Volume spike
    if vol_spike >= 1.4:
        score_long += 10
        score_short += 10  # spike может быть в обе стороны, учитываем как "движение"
    elif vol_spike >= 1.1:
        score_long += 5
        score_short += 5

    # ATR sanity
    if not math.isnan(a) and a > 0:
        score_long += 7
        score_short += 7

    # pick direction
    direction = "LONG" if score_long >= score_short else "SHORT"
    score = max(score_long, score_short)

    # dynamic SL/TP via ATR
    if math.isnan(a) or a <= 0:
        a = price * 0.005  # fallback 0.5%
    sl_mult = 1.3  # balanced
    risk = a * sl_mult

    if direction == "LONG":
        entry = price
        sl = entry - risk
        tp1 = entry + risk * 1.0
        tp2 = entry + risk * 2.0
        tp3 = entry + risk * 3.0
    else:
        entry = price
        sl = entry + risk
        tp1 = entry - risk * 1.0
        tp2 = entry - risk * 2.0
        tp3 = entry - risk * 3.0

    rr = 2.0  # TP2 примерно 2R, но покажем как ориентир

    return {
        "price": price,
        "direction": direction,
        "score": float(score),
        "ema20": float(e20),
        "ema50": float(e50),
        "ema200": float(e200),
        "rsi": float(r) if not math.isnan(r) else None,
        "atr": float(a),
        "vol_spike": float(vol_spike),
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "rr": rr,
        "struct": struct,
    }

# ---------------------------
# User settings (simple)
# ---------------------------
USER_TF = {}  # user_id -> interval
ALLOWED_TF = {"5m", "15m", "1h", "4h"}

def get_user_tf(user_id: int) -> str:
    return USER_TF.get(user_id, "15m")

# ---------------------------
# Commands
# ---------------------------
@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "🤖 BingX Signal Bot (base)\n\n"
        "Команды:\n"
        "• /bingxtest — тест BingX API\n"
        "• /settf 5m|15m|1h|4h — таймфрейм скана\n"
        "• /scan — топ сетапов\n\n"
        "Дальше добавим автосканер + авто-обновления TP/SL."
    )

@dp.message(F.text == "/bingxtest")
async def bingx_test(message: Message):
    try:
        symbols = bingx_get_usdt_perpetuals()
        sample = symbols[:5]
        h, l, c, v = bingx_get_klines_sync(sample[0], interval="1h", limit=200)
        await message.answer(
            "✅ BingX подключён\n\n"
            f"Контрактов найдено: <b>{len(symbols)}</b>\n"
            f"Пример: <code>{', '.join(sample)}</code>\n"
            f"Свечей по <code>{sample[0]}</code> (1h): <b>{len(c)}</b>"
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка BingX: <code>{type(e).__name__}: {str(e)[:400]}</code>")

@dp.message(F.text.startswith("/settf"))
async def set_tf(message: Message):
    parts = (message.text or "").split()
    if len(parts) != 2 or parts[1] not in ALLOWED_TF:
        await message.answer("Использование: <code>/settf 5m</code> или <code>/settf 15m</code> или <code>/settf 1h</code> или <code>/settf 4h</code>")
        return
    USER_TF[message.from_user.id] = parts[1]
    await message.answer(f"✅ Таймфрейм установлен: <b>{parts[1]}</b>")

@dp.message(F.text == "/scan")
async def scan(message: Message):
    user_id = message.from_user.id
    tf = get_user_tf(user_id)

    await message.answer(f"⏳ Сканирую BingX (USDT Perp) на <b>{tf}</b>…")

    symbols = bingx_get_usdt_perpetuals()

    # чтобы не грузить Railway: сканируем первые N, позже сделаем полноценный батч-скан
    # на следующем шаге расширим до всех 627 с кэшем и батчами
    N = min(120, len(symbols))

    results = []
    start_t = time.time()

    async def analyze_symbol(sym: str):
        try:
            h, l, c, v = await bingx_get_klines(sym, interval=tf, limit=220)
            s = score_setup(c, h, l, v)
            # фильтры balanced
            if s["score"] >= 80 and s["rr"] >= 1.8:
                results.append((sym, s))
        except Exception:
            pass

    tasks = [analyze_symbol(sym) for sym in symbols[:N]]
    await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    top = results[:5]

    took = time.time() - start_t

    if not top:
        await message.answer(f"⚠️ На {tf} не нашёл сетапов ≥80% в первых {N} парах. (t={took:.1f}s)\n\nПопробуй /settf 1h или /settf 4h и /scan.")
        return

    lines = [f"✅ Топ сетапы ({tf}) — проверено {N} пар за {took:.1f}s:\n"]
    for sym, s in top:
        rsi_txt = f"{s['rsi']:.1f}" if s["rsi"] is not None else "n/a"
        lines.append(
            f"🚨 <b>{sym}</b> — <b>{s['direction']}</b> | <b>{s['score']:.0f}%</b>\n"
            f"Entry: <code>{s['entry']:.6g}</code>\n"
            f"SL: <code>{s['sl']:.6g}</code>\n"
            f"TP1: <code>{s['tp1']:.6g}</code>  TP2: <code>{s['tp2']:.6g}</code>  TP3: <code>{s['tp3']:.6g}</code>\n"
            f"EMA20/50/200: <code>{s['ema20']:.6g}</code>/<code>{s['ema50']:.6g}</code>/<code>{s['ema200']:.6g}</code>\n"
            f"RSI: <code>{rsi_txt}</code> | VolSpike: <code>{s['vol_spike']:.2f}x</code> | Struct: <code>{s['struct']}</code>\n"
        )

    lines.append("⚠️ Не финсовет.")
    await message.answer("\n".join(lines))

async def main():
    print("✅ Bot is starting polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
