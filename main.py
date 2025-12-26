import os
import json
import time
import math
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

# ============================================================
#                    AYARLAR (Kolay)
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

OKX_BASE = "https://www.okx.com"
UA = {"User-Agent": "okx-spot-trend-volume-bot/1.0"}

# Tarama evreni
QUOTE = "USDT"
MAX_COINS_TO_SCAN = 150          # OKX USDT spot içinden en likit ilk N
EXCLUDE_STABLES = True

# Mum ayarları
CANDLE_LIMIT = 200               # EMA100 için yeterli
BAR_1H = "1H"
BAR_4H = "4H"

# Filtreler (güvenli sistem)
VOLRATIO_MIN_1H = 1.5            # 1H hacim en az 1.5x (20 ort.)
RSI_MIN = 40
RSI_MAX = 60

# Pullback yakınlığı (EMA20/EMA50'ye)
PULLBACK_PCT_MAX = 1.5 / 100.0   # fiyat EMA'ya %1.5 içinde olmalı

# Trend koşulları (EMA dizilimi)
USE_EMA100 = True

# Skor eşikleri
SCORE_MIN = 78

# Alarm kontrol
STATE_PATH = ".cache/state.json"
COOLDOWN_HOURS = 18              # aynı coin tekrar mesaj atmadan önce
DAILY_ALERT_LIMIT = 6            # günde max sinyal
HTTP_TIMEOUT = 12

# Risk yönetimi - TP/SL
SL_BUFFER_PCT = 0.4 / 100.0      # swing low altına ekstra buffer
TP1_R_MULT = 1.5                 # TP1 = 1.5R
TP2_R_MULT = 2.5                 # TP2 = 2.5R
MIN_R_PCT = 1.2 / 100.0          # SL mesafesi çok küçükse (noise) minimum risk bandı

# ============================================================
#                      TELEGRAM
# ============================================================

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("\n[UYARI] TELEGRAM_TOKEN veya CHAT_ID yok. Mesaj aşağıda:\n")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            print("[HATA] Telegram gönderim başarısız:", r.status_code, r.text[:300])
    except Exception as e:
        print("[HATA] Telegram exception:", str(e))

# ============================================================
#                      STATE
# ============================================================

def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso_now_tr() -> str:
    # TR saat için sadece yazı formatı
    tr = utc_now().astimezone(timezone(timedelta(hours=3)))
    return tr.strftime("%Y-%m-%d %H:%M TR")

# ============================================================
#                      OKX API
# ============================================================

def okx_get(path: str, params: Optional[dict] = None) -> dict:
    url = OKX_BASE + path
    r = requests.get(url, params=params or {}, headers=UA, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_spot_instruments_usdt() -> List[str]:
    # OKX instruments list
    data = okx_get("/api/v5/public/instruments", {"instType": "SPOT"})
    inst = []
    for row in data.get("data", []):
        instId = row.get("instId", "")
        if not instId.endswith(f"-{QUOTE}"):
            continue
        # bazıları delisted olabilir; okx yine döndürebiliyor - sorun değil
        inst.append(instId)
    return inst

def get_spot_tickers() -> List[dict]:
    data = okx_get("/api/v5/market/tickers", {"instType": "SPOT"})
    return data.get("data", [])

def get_candles(instId: str, bar: str, limit: int = 200) -> List[List[str]]:
    # OKX candles: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    data = okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
    return data.get("data", [])

# ============================================================
#                INDICATORS (EMA, RSI, ATR)
# ============================================================

def ema(values: List[float], period: int) -> List[float]:
    if len(values) < period + 1:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out

def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 2:
        return float("nan")
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return float("nan")
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return float("nan")
    return sum(trs[-period:]) / period

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ============================================================
#               PIVOT / STRUCTURE (Basit)
# ============================================================

def find_last_two_pivot_lows(lows: List[float], lookback: int = 3) -> List[Tuple[int, float]]:
    pivots = []
    # local min: low[i] < low[i-1..i-lookback] and low[i] < low[i+1..i+lookback]
    for i in range(lookback, len(lows) - lookback):
        left = lows[i - lookback:i]
        right = lows[i + 1:i + 1 + lookback]
        if lows[i] < min(left) and lows[i] < min(right):
            pivots.append((i, lows[i]))
    return pivots[-2:]  # son iki pivot

def find_last_two_pivot_highs(highs: List[float], lookback: int = 3) -> List[Tuple[int, float]]:
    pivots = []
    for i in range(lookback, len(highs) - lookback):
        left = highs[i - lookback:i]
        right = highs[i + 1:i + 1 + lookback]
        if highs[i] > max(left) and highs[i] > max(right):
            pivots.append((i, highs[i]))
    return pivots[-2:]

def structure_hl_hh(highs: List[float], lows: List[float]) -> Tuple[bool, bool]:
    low_p = find_last_two_pivot_lows(lows)
    high_p = find_last_two_pivot_highs(highs)
    hl = False
    hh = False
    if len(low_p) == 2 and low_p[-1][1] > low_p[-2][1]:
        hl = True
    if len(high_p) == 2 and high_p[-1][1] > high_p[-2][1]:
        hh = True
    return hl, hh

# ============================================================
#                 TREND / SETUP / SCORING
# ============================================================

def trend_up(closes: List[float]) -> Tuple[bool, float, float, float]:
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    e100 = ema(closes, 100) if USE_EMA100 else None

    if not e20 or not e50 or (USE_EMA100 and not e100):
        return False, float("nan"), float("nan"), float("nan")

    c = closes[-1]
    ema20 = e20[-1]
    ema50 = e50[-1]
    ema100 = e100[-1] if USE_EMA100 else float("nan")

    cond = (ema20 > ema50) and (c > ema20)
    if USE_EMA100:
        cond = cond and (ema50 > ema100)

    return cond, ema20, ema50, ema100

def volume_ratio(vols: List[float], window: int = 20) -> float:
    if len(vols) < window + 1:
        return float("nan")
    avg = sum(vols[-window-1:-1]) / window
    if avg <= 0:
        return float("nan")
    return vols[-1] / avg

def pullback_near(price: float, ema20: float, ema50: float) -> bool:
    d20 = abs(price - ema20) / ema20
    d50 = abs(price - ema50) / ema50
    return min(d20, d50) <= PULLBACK_PCT_MAX

def market_environment_btc(btc_1h: dict, btc_4h: dict) -> Tuple[str, str]:
    """
    returns (env_label, reason)
    env_label: 'GÜVENLİ' | 'KARARSIZ' | 'RİSKLİ'
    """
    # 4H trend ana filtre
    if btc_4h["trend_up"] and btc_1h["trend_up"] and btc_1h["vol_ratio"] >= 1.0:
        return "GÜVENLİ", "BTC 4H+1H UP ve hacim normal/iyi"
    if btc_4h["trend_up"] and (not btc_1h["trend_up"] or btc_1h["vol_ratio"] < 1.0):
        return "KARARSIZ", "BTC 4H UP ama 1H zayıf / hacim düşük"
    return "RİSKLİ", "BTC 4H trend DOWN veya yapı bozuk"

def score_coin(coin4h: dict, coin1h: dict) -> int:
    score = 0
    # Trend
    if coin4h["trend_up"]:
        score += 30
    if coin1h["trend_up"]:
        score += 15

    # Structure
    if coin1h["hl"]:
        score += 15
    if coin1h["hh"]:
        score += 5

    # Volume
    vr = coin1h["vol_ratio"]
    if vr >= 2.5:
        score += 25
    elif vr >= 1.8:
        score += 20
    elif vr >= 1.5:
        score += 15

    # RSI sweet spot
    r = coin1h["rsi"]
    if RSI_MIN <= r <= RSI_MAX:
        score += 15
    elif 35 <= r <= 70:
        score += 8

    # Pullback
    if coin1h["pullback"]:
        score += 10

    return min(score, 100)

def estimate_plan(instId: str, price: float, lows_1h: List[float], atr_1h: float) -> dict:
    """
    Entry zone: price etrafı küçük band
    SL: son swing low altı + buffer; yoksa son 20 low altı
    TP: R multiple
    """
    # entry zone +-0.7%
    entry_low = price * (1 - 0.007)
    entry_high = price * (1 + 0.007)

    # SL için pivot low
    pivots = find_last_two_pivot_lows(lows_1h)
    if pivots:
        swing_low = pivots[-1][1]
    else:
        swing_low = min(lows_1h[-20:]) if len(lows_1h) >= 20 else min(lows_1h)

    sl = swing_low * (1 - SL_BUFFER_PCT)

    # Minimum risk bandı (çok dipteyse)
    risk = price - sl
    min_risk = price * MIN_R_PCT
    if risk < min_risk:
        sl = price - min_risk
        risk = min_risk

    tp1 = price + TP1_R_MULT * risk
    tp2 = price + TP2_R_MULT * risk

    # ATR’e göre aşırı uçları yumuşat (opsiyon)
    # ATR çok büyükse (volatilite) tp'leri biraz yakınlaştır
    if not math.isnan(atr_1h) and atr_1h > 0:
        cap = 4.0 * atr_1h
        tp1 = min(tp1, price + cap)
        tp2 = min(tp2, price + 1.6 * cap)

    return {
        "entry_zone": (entry_low, entry_high),
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": (risk / price) * 100.0
    }

# ============================================================
#                  DATA PARSING
# ============================================================

def parse_candles(candles: List[List[str]]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    # OKX candles newest->oldest gelir, ters çeviriyoruz
    candles = list(reversed(candles))
    ts = []
    o = []
    h = []
    l = []
    c = []
    v = []
    for row in candles:
        ts.append(safe_float(row[0]))
        o.append(safe_float(row[1]))
        h.append(safe_float(row[2]))
        l.append(safe_float(row[3]))
        c.append(safe_float(row[4]))
        v.append(safe_float(row[5]))
    return o, h, l, c, v

def format_num(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

# ============================================================
#                    ALERT LOGIC
# ============================================================

def should_alert(state: dict, instId: str) -> bool:
    now = utc_now()
    day_key = now.strftime("%Y-%m-%d")

    # daily limit
    daily = state.get("daily", {})
    if daily.get("day") != day_key:
        daily = {"day": day_key, "count": 0}
    if daily.get("count", 0) >= DAILY_ALERT_LIMIT:
        state["daily"] = daily
        return False

    # cooldown per coin
    last = state.get("last_alert", {}).get(instId)
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if now - last_dt < timedelta(hours=COOLDOWN_HOURS):
                return False
        except Exception:
            pass
    return True

def mark_alert(state: dict, instId: str):
    now = utc_now()
    day_key = now.strftime("%Y-%m-%d")

    daily = state.get("daily", {})
    if daily.get("day") != day_key:
        daily = {"day": day_key, "count": 0}
    daily["count"] = daily.get("count", 0) + 1
    state["daily"] = daily

    state.setdefault("last_alert", {})
    state["last_alert"][instId] = now.isoformat()

# ============================================================
#                    MAIN RUN
# ============================================================

def main():
    state = load_state()

    # --- OKX Evren: USDT spot + tickers sıralama
    inst = set(get_spot_instruments_usdt())
    tickers = get_spot_tickers()

    # Likidite proxy: volCcy24h veya vol24h. OKX'te volCcy24h base volume gibi.
    # Stable'ları çıkaralım
    stable_like = {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDP", "PYUSD", "EUR", "EURT"}

    universe = []
    for t in tickers:
        instId = t.get("instId", "")
        if instId not in inst:
            continue
        if not instId.endswith(f"-{QUOTE}"):
            continue
        base = instId.split("-")[0]
        if EXCLUDE_STABLES and base in stable_like:
            continue
        vol = safe_float(t.get("volCcy24h", t.get("vol24h", 0)))
        last = safe_float(t.get("last", 0))
        if last <= 0:
            continue
        universe.append((instId, vol))

    universe.sort(key=lambda x: x[1], reverse=True)
    universe = universe[:MAX_COINS_TO_SCAN]

    # --- BTC durumunu çek
    btc_1h = analyze_symbol("BTC-USDT", BAR_1H)
    btc_4h = analyze_symbol("BTC-USDT", BAR_4H)
    env, env_reason = market_environment_btc(btc_1h, btc_4h)

    # Eğer RİSKLİ ortam: sadece çok seçici çalış (yine de tamamen kapatmıyorum, ama eşik yükseltiyorum)
    score_min_dynamic = SCORE_MIN + (8 if env == "RİSKLİ" else 0) - (3 if env == "GÜVENLİ" else 0)

    candidates = []

    for instId, _vol in universe:
        try:
            c1h = analyze_symbol(instId, BAR_1H)
            c4h = analyze_symbol(instId, BAR_4H)

            # Ana filtre: 4H trend up şart
            if not c4h["trend_up"]:
                continue

            # Setup filtreleri (1H)
            if not c1h["pullback"]:
                continue
            if not (RSI_MIN <= c1h["rsi"] <= RSI_MAX):
                continue
            if c1h["vol_ratio"] < VOLRATIO_MIN_1H:
                continue
            if not c1h["hl"]:
                # HL yoksa güvenli sistemde pas geç
                continue

            score = score_coin(c4h, c1h)
            if score < score_min_dynamic:
                continue

            plan = estimate_plan(instId, c1h["price"], c1h["lows"], c1h["atr"])
            candidates.append((score, instId, c1h, c4h, plan))

        except Exception as e:
            # tek coin hata verdi diye bot düşmesin
            continue

    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        # İstersen burada "bugün uygun sinyal yok" mesajı atabiliriz; spam olmasın diye kapalı tutuyorum
        save_state(state)
        return

    # En iyi 1-2 coin gönderelim (spam olmasın)
    to_send = candidates[:2]

    for score, instId, c1h, c4h, plan in to_send:
        if not should_alert(state, instId):
            continue

        msg = build_message(env, env_reason, btc_1h, btc_4h, instId, score, c1h, c4h, plan)
        send_telegram(msg)
        mark_alert(state, instId)
        time.sleep(1.2)

    save_state(state)

def analyze_symbol(instId: str, bar: str) -> dict:
    candles = get_candles(instId, bar, CANDLE_LIMIT)
    if not candles or len(candles) < 120:
        raise ValueError("Not enough candles")

    o, h, l, c, v = parse_candles(candles)

    t_up, e20, e50, e100 = trend_up(c)
    vr = volume_ratio(v, 20)
    r = rsi(c, 14)
    atrv = atr(h, l, c, 14)

    hl, hh = structure_hl_hh(h, l)
    price = c[-1]

    pull = False
    if not math.isnan(e20) and not math.isnan(e50):
        pull = pullback_near(price, e20, e50)

    return {
        "instId": instId,
        "bar": bar,
        "price": price,
        "trend_up": bool(t_up),
        "ema20": e20,
        "ema50": e50,
        "ema100": e100,
        "vol_ratio": vr if not math.isnan(vr) else 0.0,
        "rsi": r if not math.isnan(r) else 0.0,
        "atr": atrv if not math.isnan(atrv) else float("nan"),
        "hl": hl,
        "hh": hh,
        "pullback": pull,
        "highs": h,
        "lows": l,
        "closes": c,
        "vols": v,
    }

def build_message(env: str, env_reason: str, btc_1h: dict, btc_4h: dict,
                  instId: str, score: int, c1h: dict, c4h: dict, plan: dict) -> str:

    # BTC detay
    btc_line = (
        f"BTC 4H: {'🟢 UP' if btc_4h['trend_up'] else '🔴 DOWN'} | "
        f"BTC 1H: {'🟢 UP' if btc_1h['trend_up'] else '🟡/🔴 ZAYIF'} | "
        f"BTC 1H hacim: {btc_1h['vol_ratio']:.2f}x"
    )

    # Coin detay
    hlhh = []
    if c1h["hl"]:
        hlhh.append("HL")
    if c1h["hh"]:
        hlhh.append("HH")
    structure_txt = " | ".join(hlhh) if hlhh else "zayıf"

    entry_low, entry_high = plan["entry_zone"]

    # TP/SL yüzdeleri
    sl_pct = pct(plan["sl"], c1h["price"])  # negatif çıkar
    tp1_pct = pct(plan["tp1"], c1h["price"])
    tp2_pct = pct(plan["tp2"], c1h["price"])

    # Ortam emoji
    env_emoji = {"GÜVENLİ": "🟢", "KARARSIZ": "🟡", "RİSKLİ": "🔴"}.get(env, "🟡")

    txt = []
    txt.append(f"{env_emoji} SPOT TREND + HACİM SİNYALİ ({env} ORTAM)")
    txt.append(f"🕒 {iso_now_tr()} | OKX Spot | TF: 4H trend / 1H entry")
    txt.append("")
    txt.append("━━━━━━━━━━━━━━━━━━")
    txt.append("🌍 PİYASA DURUMU (BTC FİLTRESİ)")
    txt.append(btc_line)
    txt.append(f"⚠️ Ortam yorumu: {env_reason}")
    if env == "KARARSIZ":
        txt.append("📌 Not: Pozisyon küçült + sadece TP1 odak, yeni trade sayısını azalt.")
    if env == "RİSKLİ":
        txt.append("⛔ Not: Yeni alımları kıs / çok seçici ol. Stop asla silinmez.")
    txt.append("")
    txt.append("━━━━━━━━━━━━━━━━━━")
    txt.append(f"🪙 COIN: {instId} | Güven Skoru: {score}/100")
    txt.append("")
    txt.append("✅ COIN TREND (4H)")
    txt.append(f"• 4H Trend: {'🟢 UP' if c4h['trend_up'] else '🔴 DOWN'} (EMA20>EMA50{' >EMA100' if USE_EMA100 else ''} & fiyat EMA20 üstü)")
    txt.append("")
    txt.append("✅ ENTRY SETUP (1H)")
    txt.append(f"• 1H Trend: {'🟢 UP' if c1h['trend_up'] else '🟡/🔴 ZAYIF'}")
    txt.append(f"• Yapı: {structure_txt}")
    txt.append(f"• RSI(1H): {c1h['rsi']:.1f} (hedef {RSI_MIN}-{RSI_MAX})")
    txt.append(f"• Hacim(1H): {c1h['vol_ratio']:.2f}x (min {VOLRATIO_MIN_1H}x)")
    txt.append(f"• Pullback: {'✅ EMA20/EMA50 yakın' if c1h['pullback'] else '❌ uzak'}")
    txt.append("")
    txt.append("🎯 PLAN (BOT TAHMİNİ) — Spot")
    txt.append(f"📍 Entry Zone: {format_num(entry_low)} – {format_num(entry_high)}")
    txt.append(f"🛑 Stop-Loss: {format_num(plan['sl'])} ({sl_pct:.2f}%)")
    txt.append(f"🎯 TP1: {format_num(plan['tp1'])} (+{tp1_pct:.2f}%)  → %50 kâr al")
    txt.append(f"🎯 TP2: {format_num(plan['tp2'])} (+{tp2_pct:.2f}%)  → kalan kısım")
    txt.append("🧲 Opsiyon: TP1 sonrası EMA20 (1H) altı kapanışta çık (trail).")
    txt.append("")
    txt.append("✅ Özet:")
    txt.append("BTC ortam filtresi + coin 4H UP + 1H pullback + hacim onayı → spot entry adayı.")
    txt.append("Kural: Stop silinmez. BTC 1H bozulursa yeni girişler durdur.")

    return "\n".join(txt)

if __name__ == "__main__":
    main()
