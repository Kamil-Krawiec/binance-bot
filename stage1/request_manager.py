# src/data/externals/request_manager.py
import os, threading, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests

_AV_URL = (
    "https://www.alphavantage.co/query"
    "?function=TIME_SERIES_DAILY&symbol={sym}&outputsize=full&apikey={key}"
)
_FNG_URL = "https://api.alternative.me/fng/?limit=0"


class RequestManager:
    """
    Handles external contextual data (S&P-500, Fear-Greed, …) with
    on-disk caching and polite API usage.
    """

    def __init__(self, api_key: str | None, cache_dir: str = "./cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mem_cache = {}  # symbol -> DataFrame
        self._fng_daily = None  # DataFrame
        self._lock = threading.Lock()

        self.session = None
        if api_key:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": "xabc-pipeline/1.0"})

    # ------------------------------------------------------------------ #
    #  S&P-500 / market index
    # ------------------------------------------------------------------ #
    def fetch_sp500_daily(self, symbol: str = "SPY") -> pd.DataFrame:
        """Return DataFrame of daily OHLC for the symbol (cached)."""
        if symbol in self._mem_cache:
            return self._mem_cache[symbol]

        cache_file = self.cache_dir / f"{symbol}_daily.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=[0])
            df.sort_index(inplace=True)
            self._mem_cache[symbol] = df
            return df

        if not self.api_key or self.session is None:
            raise RuntimeError("Alpha Vantage API key missing; set ALPHAVANTAGE_KEY to enable macro fetches.")

        # --- API call (Alpha Vantage) ---------------------------------- #
        url = _AV_URL.format(sym=symbol, key=self.api_key)
        r = self.session.get(url, timeout=(5, 15))
        data = r.json()

        if "Time Series (Daily)" not in data:  # rate-limit?
            raise RuntimeError(f"Alpha Vantage error: {data.get('Note', data)}")

        df = (
            pd.DataFrame
            .from_dict(data["Time Series (Daily)"], orient="index")
            .astype(float)
        )
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        with self._lock:
            df.to_csv(cache_file)

        self._mem_cache[symbol] = df
        return df

    def sp500_close_on(self, ts: pd.Timestamp, symbol: str = "SPY") -> float | None:
        """Nearest prior close for `ts` (trading calendar aware)."""
        df = self.fetch_sp500_daily(symbol)
        dt = pd.to_datetime(ts).normalize()
        while dt not in df.index:
            dt -= timedelta(days=1)
            if dt < df.index[0]:
                return None
        return df.at[dt, "4. close"]

    # ------------------------------------------------------------------ #
    #  Fear & Greed Index
    # ------------------------------------------------------------------ #
    def _load_fng(self) -> pd.DataFrame:
        """
        Return a daily DataFrame with columns:
            fng              : float   (0-100)
            classification   : str     ('Fear', 'Neutral', ...)
        Cached in memory and on disk.
        """
        if self._fng_daily is not None:
            return self._fng_daily

        cache_file = self.cache_dir / "fear_greed_daily.csv"
        if cache_file.exists():
            self._fng_daily = pd.read_csv(cache_file, index_col=0, parse_dates=[0])
            return self._fng_daily

        if self.session is None:
            raise RuntimeError("Fear-Greed fetch requires network; initialize RequestManager with an API key.")

        r = self.session.get(_FNG_URL, timeout=(5, 15))
        r.raise_for_status()
        raw = r.json()["data"]

        df = (
            pd.DataFrame(raw)
              .assign(date=lambda d: pd.to_datetime(d["timestamp"].astype(int), unit="s").dt.normalize(),
                      fng=lambda d: pd.to_numeric(d["value"], errors="coerce"),
                      classification=lambda d: d["value_classification"])
              .loc[:, ["date", "fng", "classification"]]
              .set_index("date")
        )

        with self._lock:
            df.to_csv(cache_file)

        self._fng_daily = df
        return df

    def fear_greed_on(self, ts: pd.Timestamp) -> float | None:
        df = self._load_fng()
        dt = pd.to_datetime(ts).normalize()
        for lag in range(3):  # same-day, −1d, −2d
            if dt in df.index:
                return df.at[dt, "fng"]
            dt -= timedelta(days=1)
        return None
