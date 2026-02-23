import os
import asyncio
import requests
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var not set")

from aiogram.client.default import DefaultBotProperties

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

def bingx_get_usdt_perpetuals() -> list[str]:
    """
    Returns list of USDT perpetual contract symbols.
    Common format on BingX: 'BTC-USDT'
    """
    url = f"{BINGX_BASE}/openApi/swap/v2/quote/contracts"
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()

    data = js.get("data") or []
    symbols = []
    for item in data:
        # Different responses may use slightly different keys, handle safely
        quote = item.get("quoteAsset") or item.get("quote") or item.get("quoteCurrency")
        sym = item.get("symbol")
        if sym and (quote == "USDT" or (isinstance(sym, str) and sym.endswith("USDT"))):
            symbols.append(sym)

    # de-dup preserve order
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def bingx_get_klines(symbol: str, interval: str = "1h", limit: int = 200):
    """
    interval examples: 1m, 5m, 15m, 1h, 4h
    Returns: highs, lows, closes (np arrays)
    """
    url = f"{BINGX_BASE}/openApi/swap/v2/quote/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()

    candles = js.get("data") or []
    if not candles:
        raise RuntimeError(f"No candles returned for {symbol} {interval}")

    # BingX candle often looks like:
    # [time, open, high, low, close, volume] OR dicts
    # We'll support both list and dict formats.
    highs, lows, closes = [], [], []
    for c in candles:
        if isinstance(c, (list, tuple)) and len(c) >= 5:
            high = float(c[2])
            low = float(c[3])
            close = float(c[4])
        elif isinstance(c, dict):
            high = float(c.get("high") or c.get("h"))
            low = float(c.get("low") or c.get("l"))
            close = float(c.get("close") or c.get("c"))
        else:
            continue
        highs.append(high)
        lows.append(low)
        closes.append(close)

    if not closes:
        raise RuntimeError(f"Failed to parse candles for {symbol} {interval}")

    return np.array(highs, dtype=np.float64), np.array(lows, dtype=np.float64), np.array(closes, dtype=np.float64)

# ---------------------------
# Telegram handlers
# ---------------------------
@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "🤖 BingX Signal Bot (base)\n\n"
        "Команды:\n"
        "• /bingxtest — тест BingX API\n\n"
        "Дальше добавим автосканер и сигналы."
    )

@dp.message(F.text == "/bingxtest")
async def bingx_test(message: Message):
    try:
        symbols = bingx_get_usdt_perpetuals()
        if not symbols:
            await message.answer("❌ Не нашёл ни одного USDT perpetual контракта на BingX.")
            return

        sample = symbols[:5]
        h, l, c = bingx_get_klines(sample[0], interval="1h", limit=200)

        await message.answer(
            "✅ BingX подключён\n\n"
            f"Контрактов найдено: <b>{len(symbols)}</b>\n"
            f"Пример: <code>{', '.join(sample)}</code>\n"
            f"Свечей по <code>{sample[0]}</code> (1h): <b>{len(c)}</b>"
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка BingX: <code>{type(e).__name__}: {str(e)[:400]}</code>")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
