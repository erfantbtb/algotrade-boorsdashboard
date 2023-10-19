import talib as ta
import numpy as np
import pandas as pd
from _regime_detection import *
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


class PriceAction:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.returns = data.close.pct_change()
        self.returns.dropna(inplace=True)

    def candlestick_patterns(self) -> int:
        self.data["cdl_signal"] = 0

        # bullish candlestick patterns
        self.data["cdl_signal"] += ta.CDLHAMMER(self.data.open,
                                                self.data.high,
                                                self.data.low,
                                                self.data.close)
        self.data["cdl_signal"] += ta.CDLINVERTEDHAMMER(self.data.open,
                                                        self.data.high,
                                                        self.data.low,
                                                        self.data.close)
        self.data["cdl_signal"] += ta.CDLDRAGONFLYDOJI(self.data.open,
                                                       self.data.high,
                                                       self.data.low,
                                                       self.data.close)
        self.data["cdl_signal"] += ta.CDLMORNINGSTAR(self.data.open,
                                                     self.data.high,
                                                     self.data.low,
                                                     self.data.close)
        self.data["cdl_signal"] += ta.CDLPIERCING(self.data.open,
                                                  self.data.high,
                                                  self.data.low,
                                                  self.data.close)
        self.data["cdl_signal"] += ta.CDL3WHITESOLDIERS(self.data.open,
                                                        self.data.high,
                                                        self.data.low,
                                                        self.data.close)
        self.data["cdl_signal"] += ta.CDL3LINESTRIKE(self.data.open,
                                                     self.data.high,
                                                     self.data.low,
                                                     self.data.close)
        self.data["cdl_signal"] += ta.CDLMORNINGDOJISTAR(self.data.open,
                                                         self.data.high,
                                                         self.data.low,
                                                         self.data.close)

        # both candlestick patterns
        self.data["cdl_signal"] += ta.CDLABANDONEDBABY(self.data.open,
                                                       self.data.high,
                                                       self.data.low,
                                                       self.data.close)
        self.data["cdl_signal"] += ta.CDLKICKING(self.data.open,
                                                 self.data.high,
                                                 self.data.low,
                                                 self.data.close)
        self.data["cdl_signal"] += ta.CDLENGULFING(self.data.open,
                                                   self.data.high,
                                                   self.data.low,
                                                   self.data.close)
        self.data["cdl_signal"] += ta.CDLHARAMI(self.data.open,
                                                self.data.high,
                                                self.data.low,
                                                self.data.close)
        self.data["cdl_signal"] += ta.CDLSPINNINGTOP(self.data.open,
                                                     self.data.high,
                                                     self.data.low,
                                                     self.data.close)

        # bearish candlestick patterns
        self.data["cdl_signal"] += ta.CDLHANGINGMAN(self.data.open,
                                                    self.data.high,
                                                    self.data.low,
                                                    self.data.close)
        self.data["cdl_signal"] += ta.CDLSHOOTINGSTAR(self.data.open,
                                                      self.data.high,
                                                      self.data.low,
                                                      self.data.close)
        self.data["cdl_signal"] += ta.CDLGRAVESTONEDOJI(self.data.open,
                                                        self.data.high,
                                                        self.data.low,
                                                        self.data.close)
        self.data["cdl_signal"] += ta.CDLDARKCLOUDCOVER(self.data.open,
                                                        self.data.high,
                                                        self.data.low,
                                                        self.data.close)
        self.data["cdl_signal"] += ta.CDL3BLACKCROWS(self.data.open,
                                                     self.data.high,
                                                     self.data.low,
                                                     self.data.close)
        self.data["cdl_signal"] += ta.CDLEVENINGDOJISTAR(self.data.open,
                                                         self.data.high,
                                                         self.data.low,
                                                         self.data.close)
        self.data["cdl_signal"] += ta.CDLEVENINGSTAR(self.data.open,
                                                     self.data.high,
                                                     self.data.low,
                                                     self.data.close)
        return self.data["cdl_signal"].iloc[-1]

    def trend_detection(self, lambda_value: float = 12) -> int:
        rg = RegimeDetection(self.data)
        betas = rg.trend_filtering(self.returns*100, lambda_value=lambda_value)
        tr = rg.regime_switch_series(betas)
        return tr.iloc[-1]

    def support_resistance(self,
                           num_lines: list = [3, 10],
                           normalizer: int = 1000):
        # find peaks or extremas
        data = self.data.copy()
        data = data["close"].to_numpy().flatten()

        maxima = np.array(argrelextrema(data, np.greater))
        minima = np.array(argrelextrema(data, np.less))

        extrema = np.concatenate((maxima, minima), axis=1)[0]
        extrema_prices = np.concatenate(
            (data[maxima], data[minima]), axis=1)[0]

        # filter near areas
        interval = extrema_prices[0] / normalizer
        num_peaks = -999
        bandwidth = interval

        while num_lines[0] > num_peaks or num_lines[1] < num_peaks:
            initial_price = extrema_prices[0]
            kde = KernelDensity(kernel="gaussian",
                                bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))
            a, b = np.min(extrema_prices), np.max(extrema_prices)
            price_range = np.linspace(a, b, 1000).reshape(-1, 1)
            pdf = np.exp(kde.score_samples(price_range))
            peaks = find_peaks(pdf)[0]

            num_peaks = len(peaks)
            bandwidth += interval

        return price_range[peaks]


class ModernTechnicalAnalysis:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.returns = data.close.pct_change()
        self.returns.dropna(inplace=True)

    def cross_ma(self,
                 timeperiod_low: int = 20,
                 timeperiod_high: int = 80) -> int:
        ma_low = ta.SMA(self.data.close, timeperiod=timeperiod_low)[-1]
        ma_high = ta.SMA(self.data.close, timeperiod=timeperiod_high)[-1]

        if ma_high > ma_low:
            return -1

        else:
            return 1

    def rsi_simple(self,
                   threshold_up: int = 70,
                   threshold_down: int = 30,
                   timeperiod: int = 14) -> int:
        rsi = ta.RSI(self.data.close, timeperiod=timeperiod)

        if rsi[-1] >= threshold_up:
            return -1

        elif rsi[-1] <= threshold_down:
            return 1

        else:
            return 0

    def roc_cross(self,
                  timeperiod_low: int = 10,
                  timeperiod_high: int = 30,
                  threshold: float = 0) -> int:
        roc_low = ta.ROC(self.data.close, timeperiod=timeperiod_low)[-1]
        roc_high = ta.ROC(self.data.close, timeperiod=timeperiod_high)[-1]

        if roc_high > threshold and roc_low > 0:
            return 1

        elif roc_high < threshold and roc_low < 0:
            return -1

        else:
            return 0

    def mfi_simple(self,
                   threshold_up: int = 80,
                   threshold_down: int = 20,
                   timeperiod: int = 26) -> int:
        mfi = ta.MFI(self.data.high,
                     self.data.low,
                     self.data.close,
                     self.data.volume)

        if mfi[-1] >= threshold_up:
            return -1

        elif mfi[-1] <= threshold_down:
            return 1

        else:
            return 0

    def macd_simple(self,
                    fastperiod: int = 12,
                    slowperiod: int = 26,
                    signalperiod: int = 9) -> int:
        macd, signal, hist = ta.MACD(self.data.close,
                                     fastperiod=fastperiod,
                                     slowperiod=slowperiod,
                                     signalperiod=signalperiod)
        if macd[-1] > signal[-1]:
            return 1

        else:
            return -1

    def bollinger_band_simple(self, timeperiod: int = 20) -> int:
        upper, middle, lower = ta.BBANDS(self.data.close,
                                         timeperiod=timeperiod)
        if self.data.close.iloc[-1] < lower.iloc[-1]:
            return 1

        elif self.data.close.iloc[-1] < upper.iloc[-1]:
            return -1

        else:
            return 0
