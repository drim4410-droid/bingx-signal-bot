import os
import asyncio
import time
import math
import requests
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

# ==========================
# CONFIG
# ==========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var not set")

# Владелец (кому бот будет слать авто-сигналы)
# Можно НЕ задавать, тогда владелец назначается командой /setme
OWNER_ID_ENV = os.getenv("OWNER_ID")  # optional

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

BINGX_BASE = "https://open-api.bingx.com"
HTTP_TIMEOUT = 12

# PRO A settings
SCAN_INTERVAL_SEC = 5 * 60       # скан каждые 5 минут
UPDATE_INTERVAL_SEC = 60         # обновление активных сигналов каждую минуту
MAX_ACTIVE_SIGNALS = 2           # чтобы не было спама
NO_REPEAT_SEC = 6 * 60 * 60      # не повторяем тикер 6 часов

# liquidity & performance
TOP_N_TARGET = 200               # хотим держать топ-200
ROTATION_BATCH = 200             # сколько символов реально проверять за цикл (равно TOP_N_TARGET)
MAX_CONCURRENT_REQUESTS = 10     # чтобы Railway не сдох и не словить лимиты
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# thresholds (строго)
MIN_SCORE = 90.0
MIN_VOL_SPIKE = 1.30
RSI_LONG_MIN, RSI_LONG_MAX = 52.0, 65.0
RSI_SHORT_MIN, RSI_SHORT_MAX = 35.0, 48.0

# ATR risk model
SL_ATR_MULT = 1.35               # balanced/strict
TP1_R = 1.0
TP2_R = 2.5                      # PRO A: минимум 2.5R до TP2
TP3_R = 4.0                      # добивка

# ==========================
# STATE (in-memory)
# ==========================
OWNER_ID: int | None = int(OWNER_ID_ENV) if OWNER_ID_ENV and OWNER_ID_ENV.isdigit() else None

# активные сигналы: key = (symbol, direction)
ACTIVE_SIGNALS: dict[str, dict] = {}

# чтобы не повторять тикер часто
LAST_SENT_AT: dict[str, float] = {}

# простая оценка ликвидности (обновляется по объёму 15m свечей)
LIQ_SCORE: dict[str, float] = {}
LAST_LIQ_REFRESH = 0.0

# ==========================
# HTTP + BINGX
# ==========================
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

def bingx_get_klines_sync(symbol: str, interval: str, limit: int = 220):
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

async def bingx_get_klines(symbol: str, interval: str, limit: int = 220):
    async with SEM:
        return await asyncio.to_thread(bingx_get_klines_sync, symbol, interval, limit)

async def bingx_get_last_price(symbol: str) -> float:
    # Лёгкий способ без отдельного тикер-эндпоинта: 1m свечи (2 штуки)
    h, l, c, v = await bingx_get_klines(symbol, "1m", limit=2)
    return float(c[-1])

# ==========================
# INDICATORS
# ==========================
def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return arr.copy()
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
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
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    a = np.mean(tr[:period])
    for i in range(period, len(tr)):
        a = (a * (period - 1) + tr[i]) / period
    return float(a)

def slope_direction(closes: np.ndarray, lookback: int = 30) -> str:
    if len(closes) < lookback + 5:
        return "range"
    seg = closes[-lookback:]
    x = np.arange(len(seg))
    slope = float(np.polyfit(x, seg, 1)[0])
    if slope > 0:
        return "bull"
    if slope < 0:
        return "bear"
    return "range"

# ==========================
# PRO A: MTF + SCORING
# ==========================
def ema_stack_ok(price: float, e20: float, e50: float, e200: float, direction: str) -> bool:
    if direction == "LONG":
        return price > e20 > e50 > e200
    return price < e20 < e50 < e200

def late_entry_filter(price: float, e20: float, r: float, direction: str) -> bool:
    # True = OK, False = поздно / перегрето
    if math.isnan(r):
        return False
    dist = (price - e20) / e20 if e20 != 0 else 0.0
    if direction == "LONG":
        if r > 70:
            return False
        if dist > 0.03:  # >3% выше EMA20 — поздно
            return False
    else:
        if r < 30:
            return False
        if dist < -0.03:  # >3% ниже EMA20
            return False
    return True

def calc_vol_spike(volumes: np.ndarray, window: int = 50) -> float:
    if len(volumes) < 10:
        return 0.0
    v_last = float(volumes[-1])
    base = float(np.mean(volumes[-window:])) if len(volumes) >= window else float(np.mean(volumes))
    if base <= 0:
        return 0.0
    return v_last / base

def build_signal(symbol: str, direction: str, tf: str, price: float, a: float) -> dict:
    if math.isnan(a) or a <= 0:
        a = price * 0.006  # fallback 0.6%
    risk = a * SL_ATR_MULT

    entry = price
    if direction == "LONG":
        sl = entry - risk
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * TP2_R
        tp3 = entry + risk * TP3_R
    else:
        sl = entry + risk
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * TP2_R
        tp3 = entry - risk * TP3_R

    return {
        "symbol": symbol,
        "direction": direction,
        "tf": tf,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "created_at": time.time(),
        "status": "OPEN",
    }

def format_signal(sig: dict, extras: dict) -> str:
    # extras may include score, rsi, ema stack, vol_spike, etc.
    score = extras.get("score", 0)
    rsi_v = extras.get("rsi", None)
    vol_spike = extras.get("vol_spike", None)

    rsi_txt = f"{rsi_v:.1f}" if isinstance(rsi_v, (int, float)) and not math.isnan(float(rsi_v)) else "n/a"
    vs_txt = f"{vol_spike:.2f}x" if isinstance(vol_spike, (int, float)) else "n/a"

    return (
        f"🚨 <b>{sig['symbol']}</b> — <b>{sig['direction']}</b>\n"
        f"⏱ <b>{sig['tf']}</b> | 🔥 <b>{score:.0f}%</b>\n\n"
        f"🎯 Entry: <code>{sig['entry']:.6g}</code>\n"
        f"🛑 SL: <code>{sig['sl']:.6g}</code>\n"
        f"🎯 TP1: <code>{sig['tp1']:.6g}</code>\n"
        f"🎯 TP2: <code>{sig['tp2']:.6g}</code>\n"
        f"🎯 TP3: <code>{sig['tp3']:.6g}</code>\n\n"
        f"RSI: <code>{rsi_txt}</code> | VolSpike: <code>{vs_txt}</code>\n"
        f"⚠️ Не финсовет."
    )

def should_skip_repeat(symbol: str) -> bool:
    last = LAST_SENT_AT.get(symbol, 0.0)
    return (time.time() - last) < NO_REPEAT_SEC

# ==========================
# LIQUIDITY (self-learning)
# ==========================
def select_top_symbols(all_symbols: list[str]) -> list[str]:
    # если ещё нет LIQ_SCORE — берём первые TOP_N_TARGET (быстрый старт)
    if not LIQ_SCORE:
        return all_symbols[:TOP_N_TARGET]

    ranked = sorted(all_symbols, key=lambda s: LIQ_SCORE.get(s, 0.0), reverse=True)
    return ranked[:TOP_N_TARGET]

# ==========================
# MARKET SCAN (PRO A)
# ==========================
async def analyze_symbol_pro_a(symbol: str) -> tuple[dict, dict] | None:
    """
    PRO A MTF:
    - 4H trend filter
    - 1H trend confirm
    - 15m entry trigger
    """
    try:
        h4_h, h4_l, h4_c, h4_v = await bingx_get_klines(symbol, "4h", 220)
        h1_h, h1_l, h1_c, h1_v = await bingx_get_klines(symbol, "1h", 220)
        m15_h, m15_l, m15_c, m15_v = await bingx_get_klines(symbol, "15m", 220)

        price = float(m15_c[-1])

        # Liquidity update (rough): sum last 96x15m volumes (~24h)
        liq = float(np.sum(m15_v[-96:])) if len(m15_v) >= 96 else float(np.sum(m15_v))
        # EMA/RSI/ATR on each TF
        def tf_pack(closes):
            e20 = float(ema(closes, 20)[-1])
            e50 = float(ema(closes, 50)[-1])
            e200 = float(ema(closes, 200)[-1]) if len(closes) >= 200 else float(ema(closes, 100)[-1])
            return e20, e50, e200

        h4_e20, h4_e50, h4_e200 = tf_pack(h4_c)
        h1_e20, h1_e50, h1_e200 = tf_pack(h1_c)
        m15_e20, m15_e50, m15_e200 = tf_pack(m15_c)

        h4_struct = slope_direction(h4_c, 40)
        h1_struct = slope_direction(h1_c, 40)

        # decide direction by strict EMA stack on BOTH 4H and 1H
        long_ok_htf = ema_stack_ok(float(h4_c[-1]), h4_e20, h4_e50, h4_e200, "LONG") and (h4_struct == "bull")
        long_ok_h1  = ema_stack_ok(float(h1_c[-1]), h1_e20, h1_e50, h1_e200, "LONG") and (h1_struct == "bull")

        short_ok_htf = ema_stack_ok(float(h4_c[-1]), h4_e20, h4_e50, h4_e200, "SHORT") and (h4_struct == "bear")
        short_ok_h1  = ema_stack_ok(float(h1_c[-1]), h1_e20, h1_e50, h1_e200, "SHORT") and (h1_struct == "bear")

        direction = None
        if long_ok_htf and long_ok_h1:
            direction = "LONG"
        elif short_ok_htf and short_ok_h1:
            direction = "SHORT"
        else:
            # нет MTF согласия
            LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
            return None

        # 15m trigger filters
        r15 = float(rsi(m15_c, 14))
        a15 = float(atr(m15_h, m15_l, m15_c, 14))
        vs15 = calc_vol_spike(m15_v, 50)

        # RSI zone strict
        if direction == "LONG":
            if not (RSI_LONG_MIN <= r15 <= RSI_LONG_MAX):
                LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
                return None
        else:
            if not (RSI_SHORT_MIN <= r15 <= RSI_SHORT_MAX):
                LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
                return None

        # Volume spike
        if vs15 < MIN_VOL_SPIKE:
            LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
            return None

        # EMA stack on 15m тоже должен быть в сторону тренда (но мягче: цена и e20/e50)
        if direction == "LONG":
            if not (price > m15_e20 > m15_e50):
                LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
                return None
        else:
            if not (price < m15_e20 < m15_e50):
                LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
                return None

        # Late entry filter
        if not late_entry_filter(price, m15_e20, r15, direction):
            LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
            return None

        # Score (простая, но строгая)
        score = 0.0
        score += 35.0  # MTF agreement base
        score += 20.0  # strict ema stack 4h/1h already satisfied
        score += 15.0  # 15m alignment satisfied
        score += min(15.0, (vs15 - 1.0) * 10.0)  # spike bonus
        # RSI center bonus
        if direction == "LONG":
            score += max(0.0, 10.0 - abs(r15 - 58.0))
        else:
            score += max(0.0, 10.0 - abs(r15 - 42.0))
        # ATR sanity
        if not math.isnan(a15) and a15 > 0:
            score += 5.0

        # clamp
        score = max(0.0, min(100.0, score))

        if score < MIN_SCORE:
            LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)
            return None

        # Build signal
        sig = build_signal(symbol, direction, "15m", price, a15)

        extras = {
            "score": score,
            "rsi": r15,
            "vol_spike": vs15,
            "liq": liq,
        }

        # Update liquidity score
        LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)

        return sig, extras

    except Exception:
        return None

async def scan_market_and_send():
    global LAST_LIQ_REFRESH

    if OWNER_ID is None:
        return

    all_symbols = bingx_get_usdt_perpetuals()
    symbols = select_top_symbols(all_symbols)

    # анализируем ровно ROTATION_BATCH (==200)
    batch = symbols[:ROTATION_BATCH]

    candidates: list[tuple[dict, dict]] = []

    async def worker(sym: str):
        res = await analyze_symbol_pro_a(sym)
        if res:
            candidates.append(res)

    await asyncio.gather(*[worker(s) for s in batch])

    if not candidates:
        return

    # сортируем по score
    candidates.sort(key=lambda x: x[1].get("score", 0.0), reverse=True)

    # отправляем максимум 1 сигнал за скан (PRO A, без спама)
    for sig, extras in candidates:
        sym = sig["symbol"]

        if should_skip_repeat(sym):
            continue

        # Если уже есть активный по этому символу — пропускаем
        if sym in ACTIVE_SIGNALS and ACTIVE_SIGNALS[sym]["status"] == "OPEN":
            continue

        # лимит активных
        open_count = sum(1 for v in ACTIVE_SIGNALS.values() if v["status"] == "OPEN")
        if open_count >= MAX_ACTIVE_SIGNALS:
            # заменим самый слабый только если новый сильно лучше
            weakest_sym = None
            weakest_score = 999.0
            for s, v in ACTIVE_SIGNALS.items():
                if v["status"] != "OPEN":
                    continue
                sc = float(v.get("score", 0.0))
                if sc < weakest_score:
                    weakest_score = sc
                    weakest_sym = s
            if weakest_sym and extras.get("score", 0.0) >= weakest_score + 5.0 and extras.get("score", 0.0) >= 94.0:
                # закрываем слабый
                ACTIVE_SIGNALS[weakest_sym]["status"] = "REPLACED"
            else:
                continue

        # сохраняем и отправляем
        sig_record = sig.copy()
        sig_record["score"] = float(extras.get("score", 0.0))
        sig_record["rsi"] = float(extras.get("rsi", 0.0))
        sig_record["vol_spike"] = float(extras.get("vol_spike", 0.0))
        sig_record["last_update"] = time.time()

        ACTIVE_SIGNALS[sym] = sig_record
        LAST_SENT_AT[sym] = time.time()

        text = format_signal(sig_record, extras)
        await bot.send_message(OWNER_ID, text)
        break

async def update_active_signals():
    if OWNER_ID is None:
        return

    # обновляем только OPEN и максимум 2 — это легко
    for sym, sig in list(ACTIVE_SIGNALS.items()):
        if sig.get("status") != "OPEN":
            continue

        try:
            price = await bingx_get_last_price(sym)
        except Exception:
            continue

        direction = sig["direction"]
        sl = float(sig["sl"])
        tp1 = float(sig["tp1"])
        tp2 = float(sig["tp2"])
        tp3 = float(sig["tp3"])

        # уже отмеченные цели
        hit1 = bool(sig.get("hit_tp1", False))
        hit2 = bool(sig.get("hit_tp2", False))
        hit3 = bool(sig.get("hit_tp3", False))

        def send(msg: str):
            return bot.send_message(OWNER_ID, msg)

        # SL / TP checks
        if direction == "LONG":
            if price <= sl:
                sig["status"] = "SL"
                await send(f"🛑 <b>{sym}</b> — SL сработал. Цена: <code>{price:.6g}</code>")
                continue
            if (not hit1) and price >= tp1:
                sig["hit_tp1"] = True
                await send(f"✅ <b>{sym}</b> — TP1 достигнут. Цена: <code>{price:.6g}</code>")
            if (not hit2) and price >= tp2:
                sig["hit_tp2"] = True
                await send(f"✅ <b>{sym}</b> — TP2 достигнут. Цена: <code>{price:.6g}</code>")
            if (not hit3) and price >= tp3:
                sig["hit_tp3"] = True
                sig["status"] = "TP3"
                await send(f"🏁 <b>{sym}</b> — TP3 достигнут. Сделка закрыта. Цена: <code>{price:.6g}</code>")
        else:
            if price >= sl:
                sig["status"] = "SL"
                await send(f"🛑 <b>{sym}</b> — SL сработал. Цена: <code>{price:.6g}</code>")
                continue
            if (not hit1) and price <= tp1:
                sig["hit_tp1"] = True
                await send(f"✅ <b>{sym}</b> — TP1 достигнут. Цена: <code>{price:.6g}</code>")
            if (not hit2) and price <= tp2:
                sig["hit_tp2"] = True
                await send(f"✅ <b>{sym}</b> — TP2 достигнут. Цена: <code>{price:.6g}</code>")
            if (not hit3) and price <= tp3:
                sig["hit_tp3"] = True
                sig["status"] = "TP3"
                await send(f"🏁 <b>{sym}</b> — TP3 достигнут. Сделка закрыта. Цена: <code>{price:.6g}</code>")

        sig["last_update"] = time.time()

# ==========================
# BACKGROUND LOOPS
# ==========================
async def scanner_loop():
    # маленькая задержка после старта
    await asyncio.sleep(3)
    while True:
        try:
            await scan_market_and_send()
        except Exception:
            pass
        await asyncio.sleep(SCAN_INTERVAL_SEC)

async def updater_loop():
    await asyncio.sleep(5)
    while True:
        try:
            await update_active_signals()
        except Exception:
            pass
        await asyncio.sleep(UPDATE_INTERVAL_SEC)

# ==========================
# TELEGRAM COMMANDS
# ==========================
@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "🤖 BingX PRO A Signals\n\n"
        "Команды:\n"
        "• /setme — назначить меня владельцем (куда слать сигналы)\n"
        "• /bingxtest — тест BingX\n"
        "• /active — активные сигналы\n"
        "• /forcescan — принудительно просканировать сейчас\n\n"
        "Режим: <b>PRO A</b> (мало сигналов, максимально точные)"
    )

@dp.message(F.text == "/setme")
async def setme(message: Message):
    global OWNER_ID
    OWNER_ID = message.from_user.id
    await message.answer("✅ Готово. Теперь авто-сигналы будут приходить сюда.")

@dp.message(F.text == "/bingxtest")
async def bingx_test(message: Message):
    try:
        syms = bingx_get_usdt_perpetuals()
        sample = syms[:5]
        h, l, c, v = bingx_get_klines_sync(sample[0], interval="1h", limit=200)
        await message.answer(
            "✅ BingX подключён\n\n"
            f"Контрактов найдено: <b>{len(syms)}</b>\n"
            f"Пример: <code>{', '.join(sample)}</code>\n"
            f"Свечей по <code>{sample[0]}</code> (1h): <b>{len(c)}</b>"
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка BingX: <code>{type(e).__name__}: {str(e)[:400]}</code>")

@dp.message(F.text == "/active")
async def active(message: Message):
    open_items = [v for v in ACTIVE_SIGNALS.values() if v.get("status") == "OPEN"]
    if not open_items:
        await message.answer("Пока нет активных сигналов.")
        return
    lines = ["📌 <b>Активные сигналы</b>:\n"]
    for s in open_items:
        lines.append(
            f"• <b>{s['symbol']}</b> {s['direction']} | score <b>{float(s.get('score',0)):.0f}%</b>\n"
            f"  Entry <code>{s['entry']:.6g}</code> | SL <code>{s['sl']:.6g}</code>\n"
            f"  TP1 <code>{s['tp1']:.6g}</code> TP2 <code>{s['tp2']:.6g}</code> TP3 <code>{s['tp3']:.6g}</code>"
        )
    await message.answer("\n".join(lines))

@dp.message(F.text == "/forcescan")
async def forcescan(message: Message):
    await message.answer("⏳ Сканирую сейчас (PRO A)…")
    await scan_market_and_send()
    await message.answer("✅ Готово. Если был найден ultra-сетап — я отправил его.")

# ==========================
# MAIN
# ==========================
async def main():
    print("✅ Bot is starting polling...")
    # запускаем фоновые задачи
    asyncio.create_task(scanner_loop())
    asyncio.create_task(updater_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
