"""
Gold Market Sentiment Analyzer for XAUUSD Trading
Fetches news, analyzes sentiment, and outputs a normalized score (-1 to +1)
Updates: Every minute via GitHub Actions
"""

import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone
import yfinance as yf
import time

# ==================== CONFIG ====================
NEWS_RSS_FEEDS = [
    "https://www.kitco.com/rss/all.xml",
    "https://www.fxstreet.com/rss/news",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.dailyfx.com/rss/gold",
    "https://www.forexlive.com/feed/news",
]

RELEVANT_KEYWORDS = [
    "gold", "xau", "xauusd", "precious metal",
    "fed", "federal reserve", "powell", "fomc",
    "inflation", "cpi", "ppi", "pce", "deflation",
    "rate", "interest rate", "rate cut", "rate hike",
    "dollar", "dxy", "usd", "greenback",
    "safe-haven", "safe haven", "haven",
    "geopolitical", "war", "conflict", "tension",
    "treasury", "yield", "bond",
    "nfp", "employment", "unemployment", "jobs",
    "recession", "crisis", "uncertainty", "volatility",
    "central bank", "stimulus", "tapering", "qe"
]

OUTPUT_FILE_PATH = "market_context.csv"
RECENCY_HOURS = 2  # Only look at news from last 2 hours (very recent)

# Weights for final score calculation
WEIGHTS = {
    "news_sentiment": 0.60,      # 60% - News headlines (most important)
    "dxy_signal": 0.20,          # 20% - Dollar strength
    "yield_signal": 0.10,        # 10% - Treasury yields
    "vix_signal": 0.10,          # 10% - Fear index
}

# ==================== CUSTOM GOLD LEXICON ====================
GOLD_LEXICON = {
    # Bullish for Gold (positive scores)
    "rate cut": 3.5, "cut rates": 3.5, "cutting rates": 3.0,
    "dovish": 3.0, "dovish fed": 3.5, "dovish stance": 3.0,
    "weak dollar": 3.2, "dollar weakness": 3.2, "weaker dollar": 3.0,
    "inflation": 2.5, "high inflation": 3.0, "rising inflation": 2.8,
    "inflation surge": 3.0, "inflation soars": 3.2,
    "geopolitical": 2.5, "geopolitical risk": 3.0, "geopolitical tension": 3.0,
    "uncertainty": 2.3, "economic uncertainty": 2.8,
    "crisis": 3.0, "financial crisis": 3.5, "banking crisis": 3.5,
    "safe-haven": 3.5, "safe haven": 3.5, "haven demand": 3.2,
    "risk-off": 2.8, "risk aversion": 2.8,
    "recession": 2.5, "recession fears": 3.0, "recession risk": 2.8,
    "war": 3.0, "conflict": 2.5, "military": 2.0,
    "stimulus": 2.5, "easing": 2.5, "accommodation": 2.3,
    "quantitative easing": 3.0, "qe": 2.8,
    "gold rally": 3.5, "gold surge": 3.5, "gold bullish": 3.2,
    "buying gold": 2.5, "gold demand": 2.5,
    
    # Bearish for Gold (negative scores)
    "rate hike": -3.5, "hike rates": -3.5, "raising rates": -3.0,
    "hawkish": -3.0, "hawkish fed": -3.5, "hawkish stance": -3.0,
    "strong dollar": -3.2, "dollar strength": -3.2, "stronger dollar": -3.0,
    "tapering": -2.5, "taper": -2.3, "tightening": -2.5,
    "risk-on": -2.8, "risk appetite": -2.5,
    "strong economy": -2.0, "robust growth": -1.8, "economic strength": -2.0,
    "dollar rally": -3.0, "dxy surge": -3.0,
    "yields rise": -2.5, "rising yields": -2.5, "higher yields": -2.3,
    "gold falls": -3.0, "gold drops": -3.0, "gold bearish": -3.2,
    "selling gold": -2.5, "gold selloff": -3.0,
}

# ==================== SENTIMENT ANALYZER ====================
def fetch_news_sentiment():
    """Fetch and analyze news headlines with custom Gold lexicon"""
    print(f"\n{'='*70}")
    print("📰 FETCHING NEWS SENTIMENT")
    print(f"{'='*70}")
    
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(GOLD_LEXICON)
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=RECENCY_HOURS)
    headlines = []
    seen = set()
    
    for feed_url in NEWS_RSS_FEEDS:
        try:
            print(f"\n🔍 Fetching: {feed_url}")
            feed = feedparser.parse(feed_url, timeout=10)
            feed_count = 0
            
            for entry in feed.entries:
                title = entry.get('title', '')
                title_lower = title.lower()
                
                # Check recency
                pub_date = entry.get('published_parsed') or entry.get('updated_parsed')
                if pub_date:
                    pub_datetime = datetime(*pub_date[:6], tzinfo=timezone.utc)
                    if pub_datetime < cutoff_time:
                        continue
                
                # Check keywords
                if not any(kw in title_lower for kw in RELEVANT_KEYWORDS):
                    continue
                
                # Remove duplicates
                if title_lower in seen:
                    continue
                seen.add(title_lower)
                
                # Analyze sentiment
                score = analyzer.polarity_scores(title)['compound']
                headlines.append((title, score))
                feed_count += 1
                
                # Show analysis
                sentiment_label = "🟢 BULLISH" if score > 0.1 else "🔴 BEARISH" if score < -0.1 else "⚪ NEUTRAL"
                print(f"   {sentiment_label} [{score:+.3f}] {title[:80]}")
            
            print(f"   ✅ Found {feed_count} relevant headlines from this feed")
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    if not headlines:
        print("\n⚠️  WARNING: No relevant headlines found in the last 2 hours.")
        print("   Using neutral score (0.0)")
        return 0.0
    
    avg_sentiment = sum(score for _, score in headlines) / len(headlines)
    
    print(f"\n{'─'*70}")
    print(f"📊 Total headlines analyzed: {len(headlines)}")
    print(f"📈 Average news sentiment: {avg_sentiment:+.4f}")
    print(f"{'─'*70}")
    
    return avg_sentiment

# ==================== MARKET INDICATORS ====================
def get_dxy_signal():
    """Dollar Index signal (inverse relationship with Gold)"""
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        
        if len(hist) < 2:
            print("⚠️  DXY: Insufficient data")
            return 0.0
        
        current = hist['Close'].iloc[-1]
        week_ago = hist['Close'].iloc[0]
        change_pct = ((current - week_ago) / week_ago) * 100
        
        # Strong dollar = bearish for Gold (negative signal)
        signal = -change_pct / 3.0
        signal = max(-1.0, min(1.0, signal))
        
        direction = "📉 Falling" if change_pct < 0 else "📈 Rising"
        print(f"💵 DXY: {current:.2f} | {direction} {abs(change_pct):.2f}% | Signal: {signal:+.3f}")
        return signal
    
    except Exception as e:
        print(f"❌ DXY Error: {e}")
        return 0.0

def get_yield_signal():
    """10-Year Treasury Yield signal"""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        
        if len(hist) < 2:
            print("⚠️  Yield: Insufficient data")
            return 0.0
        
        current = hist['Close'].iloc[-1]
        week_ago = hist['Close'].iloc[0]
        change = current - week_ago
        
        # Rising yields = bearish for Gold
        signal = -change * 0.4
        signal = max(-1.0, min(1.0, signal))
        
        direction = "📉 Falling" if change < 0 else "📈 Rising"
        print(f"📊 10Y Yield: {current:.2f}% | {direction} {abs(change):.2f}% | Signal: {signal:+.3f}")
        return signal
    
    except Exception as e:
        print(f"❌ Yield Error: {e}")
        return 0.0

def get_vix_signal():
    """VIX Fear Index signal"""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        
        if len(hist) < 1:
            print("⚠️  VIX: Insufficient data")
            return 0.0
        
        current = hist['Close'].iloc[-1]
        
        # VIX interpretation
        if current > 30:
            signal = 0.8      # Extreme fear = strong Gold demand
            level = "🔥 EXTREME FEAR"
        elif current > 25:
            signal = 0.5      # High fear
            level = "😰 HIGH FEAR"
        elif current > 20:
            signal = 0.3      # Elevated fear
            level = "😟 ELEVATED"
        elif current < 12:
            signal = -0.3     # Complacency = bearish Gold
            level = "😌 COMPLACENCY"
        elif current < 15:
            signal = -0.1     # Low fear
            level = "😊 LOW FEAR"
        else:
            signal = 0.0      # Normal
            level = "😐 NORMAL"
        
        print(f"📉 VIX: {current:.2f} | {level} | Signal: {signal:+.3f}")
        return signal
    
    except Exception as e:
        print(f"❌ VIX Error: {e}")
        return 0.0

# ==================== MAIN CALCULATION ====================
def calculate_market_context():
    """Combine all signals into final weighted score"""
    print(f"\n{'='*70}")
    print("⚙️  GOLD MARKET CONTEXT ANALYZER")
    print(f"    Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}\n")
    
    # Fetch all components
    news_score = fetch_news_sentiment()
    
    print(f"\n{'='*70}")
    print("📊 MARKET INDICATORS")
    print(f"{'='*70}")
    
    dxy_score = get_dxy_signal()
    yield_score = get_yield_signal()
    vix_score = get_vix_signal()
    
    # Weighted average
    final_score = (
        news_score * WEIGHTS["news_sentiment"] +
        dxy_score * WEIGHTS["dxy_signal"] +
        yield_score * WEIGHTS["yield_signal"] +
        vix_score * WEIGHTS["vix_signal"]
    )
    
    # Clamp between -1 and +1
    final_score = max(-1.0, min(1.0, final_score))
    
    # Determine market bias
    if final_score > 0.5:
        bias = "🟢 STRONGLY BULLISH"
    elif final_score > 0.2:
        bias = "🟢 BULLISH"
    elif final_score > -0.2:
        bias = "⚪ NEUTRAL"
    elif final_score > -0.5:
        bias = "🔴 BEARISH"
    else:
        bias = "🔴 STRONGLY BEARISH"
    
    # Display breakdown
    print(f"\n{'='*70}")
    print("📊 FINAL CALCULATION")
    print(f"{'='*70}")
    print(f"📰 News Sentiment:    {news_score:+.4f} × {WEIGHTS['news_sentiment']:.0%} = {news_score * WEIGHTS['news_sentiment']:+.4f}")
    print(f"💵 DXY Signal:        {dxy_score:+.4f} × {WEIGHTS['dxy_signal']:.0%} = {dxy_score * WEIGHTS['dxy_signal']:+.4f}")
    print(f"📊 Yield Signal:      {yield_score:+.4f} × {WEIGHTS['yield_signal']:.0%} = {yield_score * WEIGHTS['yield_signal']:+.4f}")
    print(f"📉 VIX Signal:        {vix_score:+.4f} × {WEIGHTS['vix_signal']:.0%} = {vix_score * WEIGHTS['vix_signal']:+.4f}")
    print(f"{'─'*70}")
    print(f"🎯 FINAL SCORE:       {final_score:+.4f}")
    print(f"📈 MARKET BIAS:       {bias}")
    print(f"{'='*70}\n")
    
    return final_score

def save_score(score):
    """Save score to CSV file"""
    try:
        with open(OUTPUT_FILE_PATH, 'w') as f:
            f.write(f"{score:.4f}")
        print(f"✅ Successfully wrote score to {OUTPUT_FILE_PATH}")
        return True
    except Exception as e:
        print(f"❌ ERROR writing to file: {e}")
        return False

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        score = calculate_market_context()
        
        if save_score(score):
            print("\n✅ Script completed successfully!")
            exit(0)
        else:
            print("\n❌ Script failed to save output")
            exit(1)
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Write neutral score on failure
        print("\n⚠️  Writing neutral score (0.0) as fallback...")
        save_score(0.0)
        exit(1)
