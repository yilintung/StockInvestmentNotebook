import os
import pandas as pd
import numpy as np
import datetime
import sqlite3
import time

from talib.abstract import *

##### 個股篩選 #####
# TODO : 這個版本剛完成，還在驗證中
def StockScreener() :
    # 連線資料庫
    conn = sqlite3.connect('data/stock.db')
    cursor = conn.cursor()

    # 從資料庫中載入「台股總覽 TaiwanStockInfo」
    df_stock_info = pd.read_sql("SELECT * FROM StockInfo", conn)

    stock_id_list1 = []
    # 篩選一 : 保留加權與櫃買個股，並排除排除ETF、ETN、指數、大盤、所有證券與受益證券...等類別之股票
    #  追加排除：有英文字的股票
    for idx,stock_info in df_stock_info.iterrows() :
        stock_info_id = stock_info['StockID']
        stock_info_type = stock_info['Type']
        stock_info_category = stock_info['IndustryCategory']
        if stock_info_type == 'tpex' or stock_info_type == 'twse' :
            if 'ETF' not in stock_info_category and 'ETN' not in stock_info_category and \
              stock_info_category != 'Index' and stock_info_category != '大盤' and\
              stock_info_category != '所有證券' and stock_info_category != '受益證券' and\
              stock_info_id.isdigit() is True:
                stock_id_list1.append(stock_info_id)

    # 設定開始日期與結束日期
    current_date       = datetime.datetime.today()
    daily_end_date     = current_date.strftime('%Y-%m-%d')
    daily_start_date   = current_date - datetime.timedelta(days=240)
    daily_start_date   = daily_start_date.strftime('%Y-%m-%d')

    # 讀取日Ｋ價格資料
    sql_cmd = "SELECT * FROM DailyPrice WHERE Date BETWEEN '{}' AND '{}' ORDER BY Date".format(daily_start_date,daily_end_date)
    daily_price_df = pd.read_sql( sql_cmd, conn)

    stock_id_list2 = []
    # 篩選二 : 保留股價落在10元至100元間的個股
    #  追加條件：當日成交量大於500張
    for stock_id in stock_id_list1 :
        df_prices = daily_price_df.loc[daily_price_df['StockID'] == stock_id]
        if df_prices.empty is False :
            
            # 成交量格式轉換
            df_prices           = df_prices.drop(columns=['Value'])
            df_prices['Volume'] = df_prices['Volume'].div(1000)
            df_prices['Volume'] = df_prices['Volume'].round()
            
            last_close_price = df_prices.iloc[-1]['Close']
            last_volume      = df_prices.iloc[-1]['Volume']
            if last_close_price > 10.00 and last_close_price < 100.0 and last_volume > 500 :
                stock_id_list2.append(stock_id)

    # 篩選三 : ADX指標上升穿越20且+DI > -DI
    stock_id_list3 = []
    for stock_id in stock_id_list2 :
        df_prices = daily_price_df.loc[daily_price_df['StockID'] == stock_id]
        # 日期格式轉換
        df_prices           = df_prices.drop(columns=['SerialNo','StockID'])
        df_prices['Date']   = df_prices['Date'].astype('datetime64[ns]')
        df_prices.set_index(df_prices['Date'],inplace=True)
        df_prices           = df_prices.drop(columns=['Date'])

        # 成交量格式轉換
        df_prices           = df_prices.drop(columns=['Value'])
        df_prices['Volume'] = df_prices['Volume'].div(1000)
        df_prices['Volume'] = df_prices['Volume'].round()
        
        # 日Ｋ價格資料轉換為talib格式
        df_prices_talib          = df_prices.copy()
        df_prices_talib.columns  = [ i.lower() for i in df_prices_talib.columns]
        
        # 計算ADX指標
        talib_adx = ADX( df_prices_talib, timeperiod=14)
        if talib_adx.iloc[-1] > 20 and talib_adx.iloc[-2] < 20 :
            # 計算+DI與-DI指標
            talib_plus_di = PLUS_DI( df_prices_talib, timeperiod=14)
            talib_minus_di = MINUS_DI( df_prices_talib, timeperiod=14)
            if talib_plus_di.iloc[-1] > talib_minus_di.iloc[-1] :
                stock_id_list3.append(stock_id)
    
    # 關閉資料庫
    conn.close()
    
    return stock_id_list3
