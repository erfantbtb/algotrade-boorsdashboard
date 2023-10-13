import numpy as np  
import pandas as pd 
from _measurements import *
from typing import Union


class TickerStatistics:
    def __init__(self, 
                 symbol_price: pd.DataFrame, 
                 returns: [pd.Series, pd.DataFrame]) -> None:
        pass