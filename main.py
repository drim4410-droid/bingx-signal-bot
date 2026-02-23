import os
import asyncio
import time
import math
import requests
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.keyboard import InlineKeyboardBuilder


# ==========================
# CONFIG / DEFAULTS
# ==========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var not set")

OWNER_ID_ENV = os.getenv("OWNER_ID")  # optional
DEFAULT_OWNER_ID: int | None = int(OWNER_ID_ENV) if OWNER_ID_ENV and OWNER_ID_ENV.isdigit() else None

BINGX_BASE = "https://open-api.bingx.com"
HTTP_TIMEOUT = 12

MAX_CONCURRENT_REQUESTS = 10
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

NO_REPEAT_SEC = 6 * 60 * 60  # 6h cooldown per symbol

TOP_N_TARGET = 200
ROTATION_BATCH = 200

# Risk model defaults
SL_ATR_MULT_DEFAULT = 1.35
TP1_R = 1.0
TP2_R_DEFAULT = 2.5
TP3_R = 4.0

# Strict filters defaults (PRO A)
MIN_SCORE_DEFAULT = 90.0
MIN_VOL_SPIKE_DEFAULT = 1.30

RSI_LONG_MIN_DEFAULT, RSI_LONG_MAX_DEFAULT = 52.0, 65.0
RSI_SHORT_MIN_DEFAULT, RSI_SHORT_MAX_DEFAULT = 35.0, 48.0

SCAN_INTERVAL_SEC_DEFAULT = 5 * 60
UPDATE_INTERVAL_SEC = 60

MAX_ACTIVE_SIGNALS_DEFAULT = 2

# MTF presets
MTF_PRESETS = {
    "proA": {"htf": "4h", "mid": "1h", "ltf": "15m"},
    "fast": {"htf": "1h", "mid": "15m", "ltf": "5m"},
}


# ==========================
# BOT INIT
# ==========================
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# ==========================
# STATE (in-memory)
# ==========================
OWNER_ID: int | None = DEFAULT_OWNER_ID

ACTIVE_SIGNALS: dict[str, dict] = {}
LAST_SENT_AT: dict[str, float] = {}

LIQ_SCORE: dict[str, float] = {}

# User settings: keyed by owner_id
USER_CFG: dict[int, dict] = {}

def cfg_get(owner_id: int) -> dict:
    """Get config for owner with defaults."""
    if owner_id not in USER_CFG:
        USER_CFG[owner_id] = {
            "scan_interval_sec": SCAN_INTERVAL_SEC_DEFAULT,
            "max_active": MAX_ACTIVE_SIGNALS_DEFAULT,
            "min_score": MIN_SCORE_DEFAULT,
            "min_vol_spike": MIN_VOL_SPIKE_DEFAULT,
            "rsi_long_min": RSI_LONG_MIN_DEFAULT,
            "rsi_long_max": RSI_LONG_MAX_DEFAULT,
            "rsi_short_min": RSI_SHORT_MIN_DEFAULT,
            "rsi_short_max": RSI_SHORT_MAX_DEFAULT,
            "sl_atr_mult": SL_ATR_MULT_DEFAULT,
            "tp2_r": TP2_R_DEFAULT,
            "mtf": "proA",  # preset key
            "strictness": "PRO_A",  # ULTRA / PRO_A / SOFT
        }
    return USER_CFG[owner_id]

def now() -> float:
    return time.time()

def should_skip_repeat(symbol: str) -> bool:
    last = LAST_SENT_AT.get(symbol, 0.0)
    return (now() - last) < NO_REPEAT_SEC


# ==========================
# UI (Buttons)
# ==========================
def kb_main():
    kb = InlineKeyboardBuilder()
    kb.button(text="👑 Назначить меня владельцем", callback_data="set_owner")
    kb.button(text="⚡ Сканировать сейчас", callback_data="force_scan")
    kb.button(text="📌 Активные сигналы", callback_data="show_active")
    kb.button(text="⚙️ Настройки PRO", callback_data="settings")
    kb.button(text="✅ Тест BingX", callback_data="bingx_test")
    kb.adjust(1, 2, 2)
    return kb.as_markup()

def kb_settings(owner_id: int):
    c = cfg_get(owner_id)
    kb = InlineKeyboardBuilder()

    kb.button(text=f"⏱ Частота скана: {int(c['scan_interval_sec']/60)}м", callback_data="set_scan_interval")
    kb.button(text=f"📌 Макс активных: {c['max_active']}", callback_data="set_max_active")

    kb.button(text=f"🎯 Строгость: {c['strictness']}", callback_data="set_strictness")
    kb.button(text=f"🧠 MTF: {c['mtf']} ({MTF_PRESETS[c['mtf']]['htf']}+{MTF_PRESETS[c['mtf']]['mid']}+{MTF_PRESETS[c['mtf']]['ltf']})", callback_data="set_mtf")

    kb.button(text="⬅️ Назад", callback_data="back_main")
    kb.adjust(2, 2, 1)
    return kb.as_markup()

def kb_pick_scan_interval():
    kb = InlineKeyboardBuilder()
    kb.button(text="3 мин", callback_data="scanint:180")
    kb.button(text="5 мин", callback_data="scanint:300")
    kb.button(text="10 мин", callback_data="scanint:600")
    kb.button(text="⬅️ Назад", callback_data="settings")
    kb.adjust(3, 1)
    return kb.as_markup()

def kb_pick_max_active():
    kb = InlineKeyboardBuilder()
    kb.button(text="1", callback_data="maxact:1")
    kb.button(text="2", callback_data="maxact:2")
    kb.button(text="3", callback_data="maxact:3")
    kb.button(text="⬅️ Назад", callback_data="settings")
    kb.adjust(3, 1)
    return kb.as_markup()

def kb_pick_strictness():
    kb = InlineKeyboardBuilder()
    kb.button(text="🔥 ULTRA (очень редко)", callback_data="strict:ULTRA")
    kb.button(text="✅ PRO_A (рекоменд)", callback_data="strict:PRO_A")
    kb.button(text="🟡 SOFT (чуть чаще)", callback_data="strict:SOFT")
    kb.button(text="⬅️ Назад", callback_data="settings")
    kb.adjust(1, 1, 1, 1)
    return kb.as_markup()

def kb_pick_mtf():
    kb = InlineKeyboardBuilder()
    kb.button(text="PRO A: 4H+1H+15m", callback_data="mtf:proA")
    kb.button(text="FAST: 1H+15m+5m", callback_data="mtf:fast")
    kb.button(text="⬅️ Назад", callback_data="settings")
    kb.adjust(1, 1, 1)
    return kb.as_markup()


# ==========================
# BingX API (Public)
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
    _, _, c, _ = await bingx_get_klines(symbol, "1m", 2)
    return float(c[-1])


# ==========================
# Indicators
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

def calc_vol_spike(volumes: np.ndarray, window: int = 50) -> float:
    if len(volumes) < 10:
        return 0.0
    v_last = float(volumes[-1])
    base = float(np.mean(volumes[-window:])) if len(volumes) >= window else float(np.mean(volumes))
    if base <= 0:
        return 0.0
    return v_last / base

def ema_stack_ok(price: float, e20: float, e50: float, e200: float, direction: str) -> bool:
    if direction == "LONG":
        return price > e20 > e50 > e200
    return price < e20 < e50 < e200

def late_entry_ok(price: float, e20: float, r: float, direction: str) -> bool:
    if math.isnan(r):
        return False
    dist = (price - e20) / e20 if e20 else 0.0
    if direction == "LONG":
        if r > 70:
            return False
        if dist > 0.03:
            return False
    else:
        if r < 30:
            return False
        if dist < -0.03:
            return False
    return True

def build_signal(symbol: str, direction: str, tf: str, price: float, a: float, sl_mult: float, tp2_r: float) -> dict:
    if math.isnan(a) or a <= 0:
        a = price * 0.006
    risk = a * sl_mult

    entry = price
    if direction == "LONG":
        sl = entry - risk
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * tp2_r
        tp3 = entry + risk * TP3_R
    else:
        sl = entry + risk
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * tp2_r
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
        "created_at": now(),
        "status": "OPEN",
    }

def format_signal(sig: dict, extras: dict) -> str:
    score = extras.get("score", 0.0)
    rsi_v = extras.get("rsi", None)
    vol_spike = extras.get("vol_spike", None)
    mtf = extras.get("mtf", "")

    rsi_txt = f"{float(rsi_v):.1f}" if isinstance(rsi_v, (int, float)) and not math.isnan(float(rsi_v)) else "n/a"
    vs_txt = f"{float(vol_spike):.2f}x" if isinstance(vol_spike, (int, float)) else "n/a"

    return (
        f"🚨 <b>{sig['symbol']}</b> — <b>{sig['direction']}</b>\n"
        f"🧠 MTF: <b>{mtf}</b> | 🔥 <b>{score:.0f}%</b>\n\n"
        f"🎯 Entry: <code>{sig['entry']:.6g}</code>\n"
        f"🛑 SL: <code>{sig['sl']:.6g}</code>\n"
        f"🎯 TP1: <code>{sig['tp1']:.6g}</code>\n"
        f"🎯 TP2: <code>{sig['tp2']:.6g}</code>\n"
        f"🎯 TP3: <code>{sig['tp3']:.6g}</code>\n\n"
        f"RSI: <code>{rsi_txt}</code> | VolSpike: <code>{vs_txt}</code>\n"
        f"⚠️ Не финсовет."
    )


# ==========================
# Liquidity selection
# ==========================
def select_top_symbols(all_symbols: list[str]) -> list[str]:
    if not LIQ_SCORE:
        return all_symbols[:TOP_N_TARGET]
    ranked = sorted(all_symbols, key=lambda s: LIQ_SCORE.get(s, 0.0), reverse=True)
    return ranked[:TOP_N_TARGET]


# ==========================
# PRO Scan core
# ==========================
async def analyze_symbol(owner_id: int, symbol: str) -> tuple[dict, dict] | None:
    c = cfg_get(owner_id)
    preset = MTF_PRESETS[c["mtf"]]
    htf, mid, ltf = preset["htf"], preset["mid"], preset["ltf"]

    try:
        h_h, h_l, h_c, _ = await bingx_get_klines(symbol, htf, 220)
        m_h, m_l, m_c, _ = await bingx_get_klines(symbol, mid, 220)
        l_h, l_l, l_c, l_v = await bingx_get_klines(symbol, ltf, 220)

        price = float(l_c[-1])

        # Liquidity score: sum last ~24h on LTF
        # (для 15m: 96 свечей, для 5m: 288, для 1h: 24)
        approx_24h = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}.get(ltf, 96)
        liq = float(np.sum(l_v[-approx_24h:])) if len(l_v) >= approx_24h else float(np.sum(l_v))
        LIQ_SCORE[symbol] = max(LIQ_SCORE.get(symbol, 0.0), liq)

        def pack(closes):
            e20 = float(ema(closes, 20)[-1])
            e50 = float(ema(closes, 50)[-1])
            e200 = float(ema(closes, 200)[-1]) if len(closes) >= 200 else float(ema(closes, 100)[-1])
            return e20, e50, e200

        h_e20, h_e50, h_e200 = pack(h_c)
        m_e20, m_e50, m_e200 = pack(m_c)
        l_e20, l_e50, _ = pack(l_c)

        h_struct = slope_direction(h_c, 40)
        m_struct = slope_direction(m_c, 40)

        # direction by strict stack on HTF & MID
        long_ok = ema_stack_ok(float(h_c[-1]), h_e20, h_e50, h_e200, "LONG") and (h_struct == "bull") \
                  and ema_stack_ok(float(m_c[-1]), m_e20, m_e50, m_e200, "LONG") and (m_struct == "bull")

        short_ok = ema_stack_ok(float(h_c[-1]), h_e20, h_e50, h_e200, "SHORT") and (h_struct == "bear") \
                   and ema_stack_ok(float(m_c[-1]), m_e20, m_e50, m_e200, "SHORT") and (m_struct == "bear")

        direction = "LONG" if long_ok else "SHORT" if short_ok else None
        if not direction:
            return None

        r = float(rsi(l_c, 14))
        a = float(atr(l_h, l_l, l_c, 14))
        vs = calc_vol_spike(l_v, 50)

        # strictness tuning
        strict = c["strictness"]
        min_score = c["min_score"]
        min_vs = c["min_vol_spike"]
        rLmin, rLmax = c["rsi_long_min"], c["rsi_long_max"]
        rSmin, rSmax = c["rsi_short_min"], c["rsi_short_max"]

        if strict == "ULTRA":
            min_score = max(min_score, 94.0)
            min_vs = max(min_vs, 1.50)
        elif strict == "SOFT":
            min_score = min(min_score, 86.0)
            min_vs = min(min_vs, 1.15)

        # RSI filter
        if direction == "LONG":
            if not (rLmin <= r <= rLmax):
                return None
        else:
            if not (rSmin <= r <= rSmax):
                return None

        # Volume spike
        if vs < min_vs:
            return None

        # LTF alignment
        if direction == "LONG":
            if not (price > l_e20 > l_e50):
                return None
        else:
            if not (price < l_e20 < l_e50):
                return None

        # Late-entry
        if not late_entry_ok(price, l_e20, r, direction):
            return None

        # Score
        score = 0.0
        score += 35.0  # MTF agreement base
        score += 20.0  # strict stacks already satisfied
        score += 15.0  # LTF alignment
        score += min(15.0, (vs - 1.0) * 10.0)  # spike bonus
        # RSI center bonus
        if direction == "LONG":
            score += max(0.0, 10.0 - abs(r - 58.0))
        else:
            score += max(0.0, 10.0 - abs(r - 42.0))
        if not math.isnan(a) and a > 0:
            score += 5.0
        score = max(0.0, min(100.0, score))

        if score < min_score:
            return None

        sig = build_signal(
            symbol=symbol,
            direction=direction,
            tf=ltf,
            price=price,
            a=a,
            sl_mult=c["sl_atr_mult"],
            tp2_r=c["tp2_r"],
        )

        extras = {
            "score": score,
            "rsi": r,
            "vol_spike": vs,
            "mtf": f"{preset['htf']}+{preset['mid']}+{preset['ltf']}",
        }
        return sig, extras

    except Exception:
        return None


async def scan_market_and_send():
    global OWNER_ID
    if OWNER_ID is None:
        return

    c = cfg_get(OWNER_ID)
    all_symbols = bingx_get_usdt_perpetuals()
    symbols = select_top_symbols(all_symbols)[:ROTATION_BATCH]

    candidates: list[tuple[dict, dict]] = []

    async def worker(sym: str):
        res = await analyze_symbol(OWNER_ID, sym)
        if res:
            candidates.append(res)

    await asyncio.gather(*[worker(s) for s in symbols])

    if not candidates:
        return

    candidates.sort(key=lambda x: x[1].get("score", 0.0), reverse=True)

    # PRO A: max 1 signal per scan
    for sig, extras in candidates:
        sym = sig["symbol"]
        if should_skip_repeat(sym):
            continue
        if sym in ACTIVE_SIGNALS and ACTIVE_SIGNALS[sym].get("status") == "OPEN":
            continue

        open_count = sum(1 for v in ACTIVE_SIGNALS.values() if v.get("status") == "OPEN")
        if open_count >= c["max_active"]:
            # Replace weakest only if much better (and ultra)
            weakest_sym = None
            weakest_score = 999.0
            for s, v in ACTIVE_SIGNALS.items():
                if v.get("status") != "OPEN":
                    continue
                sc = float(v.get("score", 0.0))
                if sc < weakest_score:
                    weakest_score = sc
                    weakest_sym = s
            if weakest_sym and float(extras.get("score", 0.0)) >= weakest_score + 5.0 and float(extras.get("score", 0.0)) >= 94.0:
                ACTIVE_SIGNALS[weakest_sym]["status"] = "REPLACED"
            else:
                continue

        sig_record = sig.copy()
        sig_record["score"] = float(extras.get("score", 0.0))
        sig_record["rsi"] = float(extras.get("rsi", 0.0))
        sig_record["vol_spike"] = float(extras.get("vol_spike", 0.0))
        sig_record["mtf"] = extras.get("mtf", "")
        sig_record["last_update"] = now()

        ACTIVE_SIGNALS[sym] = sig_record
        LAST_SENT_AT[sym] = now()

        await bot.send_message(OWNER_ID, format_signal(sig_record, extras))
        break


async def update_active_signals():
    if OWNER_ID is None:
        return

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

        hit1 = bool(sig.get("hit_tp1", False))
        hit2 = bool(sig.get("hit_tp2", False))
        hit3 = bool(sig.get("hit_tp3", False))

        async def send(msg: str):
            if OWNER_ID is not None:
                await bot.send_message(OWNER_ID, msg)

        if direction == "LONG":
            if price <= sl:
                sig["status"] = "SL"
                await send(f"🛑 <b>{sym}</b> — SL. Цена: <code>{price:.6g}</code>")
                continue
            if (not hit1) and price >= tp1:
                sig["hit_tp1"] = True
                await send(f"✅ <b>{sym}</b> — TP1. Цена: <code>{price:.6g}</code>")
            if (not hit2) and price >= tp2:
                sig["hit_tp2"] = True
                await send(f"✅ <b>{sym}</b> — TP2. Цена: <code>{price:.6g}</code>")
            if (not hit3) and price >= tp3:
                sig["hit_tp3"] = True
                sig["status"] = "TP3"
                await send(f"🏁 <b>{sym}</b> — TP3. Закрыто. Цена: <code>{price:.6g}</code>")
        else:
            if price >= sl:
                sig["status"] = "SL"
                await send(f"🛑 <b>{sym}</b> — SL. Цена: <code>{price:.6g}</code>")
                continue
            if (not hit1) and price <= tp1:
                sig["hit_tp1"] = True
                await send(f"✅ <b>{sym}</b> — TP1. Цена: <code>{price:.6g}</code>")
            if (not hit2) and price <= tp2:
                sig["hit_tp2"] = True
                await send(f"✅ <b>{sym}</b> — TP2. Цена: <code>{price:.6g}</code>")
            if (not hit3) and price <= tp3:
                sig["hit_tp3"] = True
                sig["status"] = "TP3"
                await send(f"🏁 <b>{sym}</b> — TP3. Закрыто. Цена: <code>{price:.6g}</code>")

        sig["last_update"] = now()


# ==========================
# BACKGROUND TASKS
# ==========================
async def scanner_loop():
    await asyncio.sleep(3)
    while True:
        try:
            if OWNER_ID is not None:
                interval = int(cfg_get(OWNER_ID)["scan_interval_sec"])
            else:
                interval = SCAN_INTERVAL_SEC_DEFAULT
            await scan_market_and_send()
        except Exception:
            pass
        await asyncio.sleep(interval)

async def updater_loop():
    await asyncio.sleep(5)
    while True:
        try:
            await update_active_signals()
        except Exception:
            pass
        await asyncio.sleep(UPDATE_INTERVAL_SEC)


# ==========================
# TELEGRAM: /start
# ==========================
@dp.message(F.text == "/start")
async def start(message: Message):
    text = (
        "🤖 <b>BingX PRO Signals</b>\n\n"
        "Управление — кнопками ниже.\n"
        "Режим: <b>PRO A</b> (мало сигналов, максимально точные)"
    )
    await message.answer(text, reply_markup=kb_main())


# ==========================
# CALLBACKS
# ==========================
@dp.callback_query(F.data == "back_main")
async def back_main(cb: CallbackQuery):
    await cb.message.edit_text(
        "🤖 <b>BingX PRO Signals</b>\n\nВыбери действие:",
        reply_markup=kb_main()
    )
    await cb.answer()

@dp.callback_query(F.data == "set_owner")
async def set_owner(cb: CallbackQuery):
    global OWNER_ID
    OWNER_ID = cb.from_user.id
    cfg_get(OWNER_ID)  # init defaults
    await cb.message.edit_text(
        "✅ Готово. Теперь авто-сигналы будут приходить сюда.\n\n"
        "Можешь нажать «⚡ Сканировать сейчас» для проверки.",
        reply_markup=kb_main()
    )
    await cb.answer("Владелец назначен ✅")

@dp.callback_query(F.data == "bingx_test")
async def bingx_test(cb: CallbackQuery):
    try:
        syms = bingx_get_usdt_perpetuals()
        sample = syms[:5]
        _, _, c, _ = bingx_get_klines_sync(sample[0], "1h", 200)
        await cb.message.edit_text(
            "✅ <b>BingX подключён</b>\n\n"
            f"Контрактов: <b>{len(syms)}</b>\n"
            f"Пример: <code>{', '.join(sample)}</code>\n"
            f"Свечей по <code>{sample[0]}</code> (1h): <b>{len(c)}</b>\n\n"
            "⬅️ Вернуться в меню — кнопкой ниже.",
            reply_markup=kb_main()
        )
    except Exception as e:
        await cb.message.edit_text(
            f"❌ Ошибка BingX: <code>{type(e).__name__}: {str(e)[:300]}</code>",
            reply_markup=kb_main()
        )
    await cb.answer()

@dp.callback_query(F.data == "force_scan")
async def force_scan(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала назначь владельца 👑", show_alert=True)
        return
    await cb.answer("Сканирую…")
    await bot.send_message(OWNER_ID, "⏳ Сканирую рынок (PRO)…")
    await scan_market_and_send()
    await bot.send_message(OWNER_ID, "✅ Готово. Если был ultra-сетап — я отправил сигнал.")

@dp.callback_query(F.data == "show_active")
async def show_active(cb: CallbackQuery):
    open_items = [v for v in ACTIVE_SIGNALS.values() if v.get("status") == "OPEN"]
    if not open_items:
        await cb.message.edit_text("Пока нет активных сигналов.", reply_markup=kb_main())
        await cb.answer()
        return

    lines = ["📌 <b>Активные сигналы</b>:\n"]
    for s in open_items:
        lines.append(
            f"• <b>{s['symbol']}</b> {s['direction']} | score <b>{float(s.get('score',0)):.0f}%</b>\n"
            f"  Entry <code>{s['entry']:.6g}</code> | SL <code>{s['sl']:.6g}</code>\n"
            f"  TP1 <code>{s['tp1']:.6g}</code>  TP2 <code>{s['tp2']:.6g}</code>  TP3 <code>{s['tp3']:.6g}</code>"
        )
    await cb.message.edit_text("\n".join(lines), reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(F.data == "settings")
async def settings(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала назначь владельца 👑", show_alert=True)
        return
    await cb.message.edit_text("⚙️ <b>Настройки PRO</b>\nВыбери что менять:", reply_markup=kb_settings(OWNER_ID))
    await cb.answer()

@dp.callback_query(F.data == "set_scan_interval")
async def set_scan_interval(cb: CallbackQuery):
    await cb.message.edit_text("⏱ Выбери частоту скана:", reply_markup=kb_pick_scan_interval())
    await cb.answer()

@dp.callback_query(F.data.startswith("scanint:"))
async def pick_scan_interval(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала /start", show_alert=True)
        return
    val = int(cb.data.split(":")[1])
    cfg_get(OWNER_ID)["scan_interval_sec"] = val
    await cb.message.edit_text("✅ Частота скана обновлена.", reply_markup=kb_settings(OWNER_ID))
    await cb.answer("Готово ✅")

@dp.callback_query(F.data == "set_max_active")
async def set_max_active(cb: CallbackQuery):
    await cb.message.edit_text("📌 Макс активных сигналов:", reply_markup=kb_pick_max_active())
    await cb.answer()

@dp.callback_query(F.data.startswith("maxact:"))
async def pick_max_active(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала /start", show_alert=True)
        return
    val = int(cb.data.split(":")[1])
    cfg_get(OWNER_ID)["max_active"] = val
    await cb.message.edit_text("✅ Лимит активных обновлён.", reply_markup=kb_settings(OWNER_ID))
    await cb.answer("Готово ✅")

@dp.callback_query(F.data == "set_strictness")
async def set_strictness(cb: CallbackQuery):
    await cb.message.edit_text("🎯 Выбери строгость:", reply_markup=kb_pick_strictness())
    await cb.answer()

@dp.callback_query(F.data.startswith("strict:"))
async def pick_strictness(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала /start", show_alert=True)
        return
    val = cb.data.split(":")[1]
    cfg = cfg_get(OWNER_ID)
    cfg["strictness"] = val

    # tune thresholds by strictness
    if val == "ULTRA":
        cfg["min_score"] = 94.0
        cfg["min_vol_spike"] = 1.50
        cfg["tp2_r"] = 3.0
    elif val == "SOFT":
        cfg["min_score"] = 86.0
        cfg["min_vol_spike"] = 1.15
        cfg["tp2_r"] = 2.2
    else:  # PRO_A
        cfg["min_score"] = 90.0
        cfg["min_vol_spike"] = 1.30
        cfg["tp2_r"] = 2.5

    await cb.message.edit_text("✅ Строгость применена.", reply_markup=kb_settings(OWNER_ID))
    await cb.answer("Ок ✅")

@dp.callback_query(F.data == "set_mtf")
async def set_mtf(cb: CallbackQuery):
    await cb.message.edit_text("🧠 Выбери MTF пресет:", reply_markup=kb_pick_mtf())
    await cb.answer()

@dp.callback_query(F.data.startswith("mtf:"))
async def pick_mtf(cb: CallbackQuery):
    if OWNER_ID is None:
        await cb.answer("Сначала /start", show_alert=True)
        return
    val = cb.data.split(":")[1]
    if val not in MTF_PRESETS:
        await cb.answer("Неизвестный пресет", show_alert=True)
        return
    cfg_get(OWNER_ID)["mtf"] = val
    await cb.message.edit_text("✅ MTF пресет обновлён.", reply_markup=kb_settings(OWNER_ID))
    await cb.answer("Готово ✅")


# ==========================
# MAIN
# ==========================
async def main():
    print("✅ Bot is starting polling...")
    asyncio.create_task(scanner_loop())
    asyncio.create_task(updater_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
