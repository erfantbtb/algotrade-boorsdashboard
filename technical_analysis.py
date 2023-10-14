import talib as ta 
import numpy as np 
import pandas as pd 
from _regime_detection import *

class TechnicalAnalysis:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data 
        self.returns = data.close.pct_change()
        self.returns.dropna(inplace=True)
    
    def candlestick_patterns(self) -> int:
        self.data["cdl_signal"] = 0
        
        #bullish candlestick patterns
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

        #both candlestick patterns
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
        
        #bearish candlestick patterns
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
        
        
        
        
        