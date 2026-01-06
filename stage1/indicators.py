import numpy as np
import pandas as pd

try:
    import ta  # type: ignore
except Exception:  # pragma: no cover - optional dep
    ta = None


class TechnicalIndicators:
    """
    Computes, stores, and provides access to a wide variety of technical indicators for OHLCV data.
    """

    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, ohlc: pd.DataFrame):
        """
        Args:
            ohlc (pd.DataFrame): Must contain columns ['open', 'high', 'low', 'close', 'volume']
        """
        missing = [c for c in self.REQUIRED_COLS if c not in ohlc.columns]
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        self.ohlc = ohlc.reset_index(drop=True)
        self.indicators = pd.DataFrame(index=self.ohlc.index)
        self._calculate_all()

    def _calculate_all(self):
        close = self.ohlc["close"]
        high = self.ohlc["high"]
        low = self.ohlc["low"]
        volume = self.ohlc["volume"]

        if ta is not None:
            # --- ta-lib based ---
            # RSI
            self.indicators['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            # MACD
            macd = ta.trend.MACD(close)
            self.indicators['MACD'] = macd.macd()
            self.indicators['MACD_signal'] = macd.macd_signal()
            self.indicators['MACD_hist'] = macd.macd_diff()
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            self.indicators['stoch_k'] = stoch.stoch()
            self.indicators['stoch_d'] = stoch.stoch_signal()
            # ADX
            adx = ta.trend.ADXIndicator(high, low, close)
            self.indicators['ADX'] = adx.adx()
            # CCI
            cci = ta.trend.CCIIndicator(high, low, close)
            self.indicators['CCI'] = cci.cci()
            # OBV
            self.indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            # EMA
            self.indicators['EMA_14'] = close.ewm(span=14, adjust=False).mean()
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            self.indicators['BB_upper'] = bb.bollinger_hband()
            self.indicators['BB_middle'] = bb.bollinger_mavg()
            self.indicators['BB_lower'] = bb.bollinger_lband()
            self.indicators['BB_bandwidth'] = bb.bollinger_wband()
            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close)
            self.indicators['ATR'] = atr.average_true_range()
            # PSAR
            psar = ta.trend.PSARIndicator(high, low, close)
            self.indicators['PSAR'] = psar.psar()
            # Volume MA and Oscillator
            self.indicators['Volume_MA_20'] = volume.rolling(window=20).mean()
            self.indicators['Volume_MA_5'] = volume.rolling(window=5).mean()
            self.indicators['Volume_Oscillator'] = ((self.indicators['Volume_MA_5'] - self.indicators['Volume_MA_20']) /
                                                    self.indicators['Volume_MA_20']) * 100
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(high, low)
            self.indicators['Ichimoku_Tenkan_Sen'] = ichimoku.ichimoku_conversion_line()
            self.indicators['Ichimoku_Kijun_Sen'] = ichimoku.ichimoku_base_line()
            self.indicators['Ichimoku_Senkou_Span_A'] = ichimoku.ichimoku_a()
            self.indicators['Ichimoku_Senkou_Span_B'] = ichimoku.ichimoku_b()
            self.indicators['Ichimoku_Chikou_Span'] = close.shift(-26)
        else:
            # --- fallback: minimal indicators ---
            self.indicators['RSI_14'] = self._rsi(close, 14)
            macd, macd_signal, macd_hist = self._macd(close)
            self.indicators['MACD'] = macd
            self.indicators['MACD_signal'] = macd_signal
            self.indicators['MACD_hist'] = macd_hist
            bb_upper, bb_middle, bb_lower = self._bollinger_bands(close)
            self.indicators['BB_upper'] = bb_upper
            self.indicators['BB_middle'] = bb_middle
            self.indicators['BB_lower'] = bb_lower
            self.indicators['EMA_14'] = self._ema(close, 14)
            self.indicators['ATR'] = (high - low).rolling(window=14, min_periods=14).mean()
            self.indicators['OBV'] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
            self.indicators['Volume_MA_20'] = volume.rolling(20).mean()
            # ...extend as needed

        # Optionally copy over open, high, low, close, volume columns for reference:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in self.indicators.columns and col in self.ohlc.columns:
                self.indicators[col] = self.ohlc[col]

    # --- Fallback manual indicator functions if 'ta' is not present ---
    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _ema(self, series, period=14):
        return series.ewm(span=period, adjust=False).mean()

    def _macd(self, series, fast=12, slow=26, signal=9):
        fast_ema = self._ema(series, fast)
        slow_ema = self._ema(series, slow)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def _bollinger_bands(self, series, period=20, std_dev=2):
        mid = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    # --- Public interface ---
    def get_indicator(self, name: str, idx: int) -> float:
        """Get a specific indicator value at row idx."""
        try:
            return float(self.indicators[name].iloc[idx])
        except Exception:
            return np.nan

    def get_all_at(self, idx: int) -> dict:
        """Get all indicator values at a given index as a dictionary."""
        try:
            return self.indicators.iloc[idx].to_dict()
        except Exception:
            return {}

    def get_row(self, idx: int) -> pd.Series:
        """Return all indicator values at index as a Series."""
        return self.indicators.iloc[idx]

    def get_column(self, name: str) -> pd.Series:
        """Get a full indicator series by name."""
        return self.indicators[name]

    def calculate_angle(self, idx1: int, idx2: int) -> float:
        """
        Compute the angle (in degrees) between two close prices.
        """
        t1, t2 = idx1, idx2
        y1, y2 = self.ohlc['close'].iloc[t1], self.ohlc['close'].iloc[t2]
        radians = np.arctan2(y2 - y1, t2 - t1)
        return np.degrees(radians)

    @staticmethod
    def hour_bin(hour: int) -> str:
        """Categorize hour into bins for feature engineering."""
        if hour < 6: return '0-5'
        if hour < 12: return '6-11'
        if hour < 18: return '12-17'
        return '18-23'
