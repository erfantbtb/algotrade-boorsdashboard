import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import pytse_client as tse 
import streamlit as st 
import tensorflow as tf 
import os 
import warnings 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _measurements import *
warnings.filterwarnings("ignore")


class StockDashboard:
    def __init__(self, name: str = "Tehran's Boors dashboard") -> None:
        self.header = st.container()
        self.symbol = None
        
        if "tickers_data" in os.listdir():
            pass
        else:
            tse.download(symbols="all", write_to_csv=True)
            
        st.title(name)
        
    def side_bar(self):
        self.symbol = st.sidebar.text_input("Symbol", self.symbol)
        self.start_date = st.sidebar.date_input("Start Date")
        self.end_date = st.sidebar.date_input("End Date")
    
    def get_data(self):
        self.price = pd.read_csv(os.path.join("tickers_data", self.symbol+".csv"),
                                 parse_dates=True, 
                                 index_col="date")
        fig_main = make_subplots(rows=2, 
                                 cols=1, 
                                 shared_xaxes=True,
                                 vertical_spacing=0.03, 
                                 subplot_titles=[self.symbol, "Volume"], 
                                 row_width=[0.2, 0.7])

        fig_main.layout.template = "plotly_dark"

        fig_main.add_trace(go.Candlestick(
            x = self.price.index,
            open = self.price.open,
            high = self.price.high,
            low = self.price.low,
            close = self.price.close,
            name=self.symbol,
            showlegend=False
        ), row=1, col=1)

        fig_main.add_trace(go.Bar(x=self.price.index,
                                y=self.price["volume"],
                                name="Volume",
                                showlegend=False),
                                row=2, 
                                col=1)

        fig_main.update_layout(width=800, height=600)
        st.plotly_chart(fig_main, use_container_width=True)
        rets = self.price["close"].pct_change()
        rets.dropna(inplace=True)
        stats = Measurements([1, 2, 3]).analyze(rets, 1)
        stats.index.name = "statistics"
        st.dataframe(stats)
                
                
if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.side_bar()
    dashboard.get_data()