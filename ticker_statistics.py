import numpy as np  
import pandas as pd 
from _measurements import *
import pytse_client as tse 
from typing import Union


class ReturnStatistic:
    def __init__(self,  
                 symbol_price: pd.DataFrame, 
                 strategy_returns: [pd.Series, pd.DataFrame, None] = None) -> None:
        self.symbol_price = symbol_price 
        
        if strategy_returns == None:
            self.strategy_returns = self.symbol_price["close"].pct_change()
            self.strategy_returns.dropna(inplace=True)   
            
        elif isinstance(strategy_returns, pd.DataFrame) or isinstance(strategy_returns, pd.Series):
            self.strategy_returns = strategy_returns 
            
        else:
            raise TypeError("strategy_returns should be either dataframe or series!")
        
    def check_data(self) -> bool:
        pass 
    
    def compute_returns_stats(self) -> pd.DataFrame: 
        stats = Measurements([1, 2, 3]).analyze(self.strategy_returns, 1)
        stats.index.name = "statistics"
        return stats 
    

class FundamentalData:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol 
        self.ticker = tse.Ticker(self.symbol)
        
    def symbol_information(self) -> None:
        self.title = self.ticker.title
        self.group_name = self.ticker.group_name 
        self.financial_year = self.ticker.fiscal_year 
        
    def fundamental_data(self) -> None:
        self.eps = self.ticker.eps 
        self.p_e = self.ticker.p_e_ratio
        self.group_p_e = self.ticker.group_p_e_ratio 
        self.market_cap = self.ticker.market_cap 
        
    def fundamental_table(self) -> pd.DataFrame:
        fundamental_dict = {
            "symbol": self.title, 
            "group": self.group_name, 
            "financial_year": self.financial_year, 
            "eps": self.eps, 
            "p_e": self.p_e, 
            "group p_e": self.group_p_e, 
            "market cap": self.market_cap
        }
        # return pd.DataFrame(fundamental_dict)
        return fundamental_dict
        
    
        
    
        
        
        
        
    
              
        
              