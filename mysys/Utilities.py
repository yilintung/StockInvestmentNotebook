import io
import os
import sys
import base64
import matplotlib.pyplot as plt
import matplotlib
import mplfinance as mpf
import datetime
import sqlite3
import requests
import time

import pandas as pd

from talib.abstract import *

from PIL import Image

from dotenv import load_dotenv, find_dotenv
from FinMind.data import DataLoader

from .StockAnalysis import get_monday_to_sunday
from .StockAnalysis import date_to_index
from .StockAnalysis import crop_borders

##### 公用函式：將日期轉換為索引 #####
# 參數：價格資料(DataFrame)
#      日期
# 回傳：索引值
def DateToIndex( prices_df, date_in) :
    if date_in not in prices_df.index :
        return -1
    return date_to_index( prices_df, date_in)

##### 公用函式：在K線圖上畫線 #####
# 參數：股票代碼
#      區間開始日期(該日期需要存在於價格資料中)
#      區間結束日期(該日期需要存在於價格資料中)
#      回呼函式
# 回傳：K線圖之圖像
# 回呼函式格式：
#  參數：價格資料
#  回傳：線段（點到點之序列）
#       線寬
#       線段顏色
def DrawOnKlineChart( stock_id, range_start_date, range_end_date, callback_function) :
    # 設定價格資料之日期範圍
    current_date       = datetime.datetime.today()
    daily_end_date     = current_date.strftime('%Y-%m-%d')
    daily_start_date,_ = get_monday_to_sunday((current_date - datetime.timedelta(days=730)).strftime('%Y-%m-%d'))

    # 連線資料庫
    conn = sqlite3.connect('data/stock.db')

    # 從資料庫中載入「台股總覽 TaiwanStockInfo」
    df_stock_info = pd.read_sql("SELECT * FROM StockInfo", conn)

    # 將載入的「台股總覽 TaiwanStockInfo」進行格式轉換
    df_stock_info.set_index(df_stock_info['StockID'],inplace=True)
    df_stock_info = df_stock_info.drop(columns=['StockID'])
    
    # 取得該股票代碼的產業分類
    industry_category = df_stock_info.loc[stock_id]['IndustryCategory']
    
    # 讀取日Ｋ價格資料
    sql_cmd = "SELECT * FROM DailyPrice WHERE StockID='{}' AND (Date BETWEEN '{}' AND '{}')".format(stock_id,daily_start_date,daily_end_date)
    daily_price_df = pd.read_sql( sql_cmd, conn)
    
    # 格式轉換：日期格式、成交量(成交值)
    daily_price_df           = daily_price_df.drop(columns=['SerialNo','StockID'])
    daily_price_df['Date']   = daily_price_df['Date'].astype('datetime64[ns]')
    daily_price_df.set_index(daily_price_df['Date'],inplace=True)
    daily_price_df.drop(columns=['Date'],inplace=True)
    if industry_category == 'Index' or industry_category == '大盤' :
        daily_price_df           = daily_price_df.drop(columns=['Volume'])
        daily_price_df           = daily_price_df.rename(columns={'Value':'Volume'})
        daily_price_df['Volume'] = daily_price_df['Volume'].div(100000000.00)
        daily_price_df['Volume'] = daily_price_df['Volume'].round(2)
    else :
        daily_price_df           = daily_price_df.drop(columns=['Value'])
        daily_price_df['Volume'] = daily_price_df['Volume'].div(1000)
        daily_price_df['Volume'] = daily_price_df['Volume'].round()
        daily_price_df['Volume'] = daily_price_df['Volume'].astype('int64')

    # 關閉資料庫
    conn.close()
    
    # 確認範圍的開始與結束日期存在於價格資料中
    if range_start_date not in daily_price_df.index or range_end_date not in daily_price_df.index :
        return None
        
    # 日Ｋ價格資料轉換為talib格式
    daily_price_df_talib          = daily_price_df.copy()
    daily_price_df_talib.columns  = [ i.lower() for i in daily_price_df_talib.columns]
            
    # 計算移動平均線
    talib_sma5        = SMA( daily_price_df_talib, timeperiod=5)
    talib_sma10       = SMA( daily_price_df_talib, timeperiod=10)
    talib_sma20       = SMA( daily_price_df_talib, timeperiod=20)
    talib_sma60       = SMA( daily_price_df_talib, timeperiod=60)
    talib_sma120      = SMA( daily_price_df_talib, timeperiod=120)
    talib_sma240      = SMA( daily_price_df_talib, timeperiod=240)
    # 設定名稱
    talib_sma5.name   = 'SMA5'
    talib_sma10.name  = 'SMA10'
    talib_sma20.name  = 'SMA20'
    talib_sma60.name  = 'SMA60'
    talib_sma120.name = 'SMA120'
    talib_sma240.name = 'SMA240'
    # 合併各條均線
    talib_sma_df      = pd.concat([talib_sma5, talib_sma10, talib_sma20, talib_sma60, talib_sma120, talib_sma240], axis=1)
    # 取小數點後兩位
    sma_df            = talib_sma_df.round(2)
    
    # 設定價格範圍
    range_prices = daily_price_df[range_start_date:range_end_date]
    # 設定均線範圍
    range_sma    = sma_df[range_start_date:range_end_date]
    
    # 呼叫回呼函式
    try:
        seq_of_seq_of_points, linewidths, colors = callback_function(range_prices)
    except TypeError:
        print('ＥＲＲＯＲ：回呼函式回傳格式有誤')
        return None
    if type(seq_of_seq_of_points) is not list or type(linewidths) is not list or type(colors) is not list :
        print('ＥＲＲＯＲ：回呼函式回傳格式有誤')
        return None
    
    # 設定K線格式
    mc = mpf.make_marketcolors(up='xkcd:light red', down='xkcd:almost black', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    # 設定均線
    added_plots = [
        mpf.make_addplot(range_sma['SMA5'],width=0.8,color='xkcd:red brown'),
        mpf.make_addplot(range_sma['SMA10'],width=0.8,color='xkcd:dark sky blue'),
        mpf.make_addplot(range_sma['SMA20'],width=1.0,color='xkcd:violet'),
        mpf.make_addplot(range_sma['SMA60'],width=0.8,color='xkcd:orange')
    ]

    # 繪製K線圖
    img = io.BytesIO()
    kwargs = dict(type='candle', style=s, figratio=(19,10), addplot=added_plots, alines=dict(alines=seq_of_seq_of_points, linewidths=tuple(linewidths), colors=tuple(colors), alpha=0.6), volume=True, datetime_format='%Y-%m-%d',savefig=dict(fname=img,format='png'))
    mpf.plot(range_prices,**kwargs)
    
    # 刪除空白
    image    = Image.open(img).convert("RGB")
    image    = crop_borders(image, crop_color=(255, 255, 255))
    crop_img = io.BytesIO()
    image.save(crop_img,format='PNG')
    
    return image
    

##### 更新技術分析資料庫公用函式之私有輔助函式 #####
# FinMind API 使用次數
# 參考：https://finmind.github.io/api_usage_count/
def api_usage( token) :
    url = "https://api.web.finmindtrade.com/v2/user_info"
    payload = {
        "token": token,
    }
    resp = requests.get(url, params=payload)
    return resp.json()["user_count"],resp.json()["api_request_limit"]

# Python 取得時間範圍內日期列表
# 來源：https://www.cnblogs.com/xiao-xue-di/p/11900649.html
def date_range( beginDate, endDate):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates
    
    
##### 公用函式：更新技術分析資料庫 #####
# 參數：回朔天數(預設為1)
#      更新日K與否旗標(預設為True)
#      更新週K與否旗標(預設為False)
#      更新月K與否旗標(預設為False)
# 回傳：無
def UpdatestockDatabase(prev_days=1, update_daily_price = True, update_weekly_price = False, update_monthly_price = False) :
    
    # 設定FinMind API
    load_dotenv(find_dotenv())
    token = os.environ.get('FINMIND_TOKEN')
    api = DataLoader()
    api.login_by_token(api_token=token)
    
    # 取得範圍日期列表
    today_date      = datetime.datetime.today()
    start_date      = today_date - datetime.timedelta(days=prev_days)
    price_date_list = date_range(start_date.strftime('%Y-%m-%d'), today_date.strftime('%Y-%m-%d'))
    
    # 連線資料庫
    conn = sqlite3.connect('data/stock.db')
    
    # 設定支援外鍵
    conn.execute('PRAGMA foreign_keys = ON;')
    
    # 從資料庫中載入「台股總覽 TaiwanStockInfo」
    df_stock_info = pd.read_sql("SELECT * FROM StockInfo", conn)
    
    # 台股總覽 TaiwanStockInfo
    df               = api.taiwan_stock_info()
    df_tw_stock_info = df.drop(columns=['date'])
    df_tw_stock_info = df_tw_stock_info.rename(columns={'stock_id':'StockID','stock_name':'StockName','industry_category':'IndustryCategory','type':'Type'})
    df_tw_stock_info.drop_duplicates(subset=['StockID'], keep='first', inplace=True)
    df_tw_stock_info = df_tw_stock_info[['StockID','StockName','IndustryCategory','Type']]
    
    # 確認「台股總覽 TaiwanStockInfo」是否更新？
    # 參考 ： https://www.learncodewithmike.com/2021/10/pandas-compare-values-between-dataframes.html
    # 參考 ： https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
    df_diff = df_stock_info.merge(df_tw_stock_info, how='outer', indicator=True)
    
    print('##### 更新台股總覽 #####')
    
    # 更新資料庫中的「台股總覽 TaiwanStockInfo」
    df_diff = df_diff.loc[lambda x : x['_merge'] == 'right_only']
    for idx in range(df_diff.shape[0]) :
        info   = df_diff.iloc[idx]
        sqlcmd = "SELECT * FROM StockInfo WHERE StockID='{}'".format(info['StockID'])
        df     = pd.read_sql_query(sqlcmd, conn)
        if df.empty is True :
            print('新增股票代碼。代碼 ＝ {}：股票名稱 ＝ {}，產業類別 ＝ {}，類型 ＝ {}'.format(info['StockID'],info['StockName'],info['IndustryCategory'],info['Type']))
            conn.execute("INSERT INTO StockInfo(StockID,StockName,IndustryCategory,Type) VALUES(?,?,?,?)", (info['StockID'],info['StockName'],info['IndustryCategory'],info['Type']))
        else :
            print('修改股票資訊。代碼 ＝ {}：股票名稱 ＝ {}，產業類別 ＝ {}，類型 ＝ {}'.format(info['StockID'],info['StockName'],info['IndustryCategory'],info['Type']))
            sqlcmd = "UPDATE StockInfo SET 'StockName'='{}','IndustryCategory'='{}','Type'='{}' WHERE StockID='{}'".format(info['StockName'],info['IndustryCategory'],info['Type'],info['StockID'])
            conn.execute(sqlcmd)

    # 修改資料庫
    conn.commit()
    
    # 從資料庫中載入「台股總覽 TaiwanStockInfo」
    df_stock_info = pd.read_sql("SELECT * FROM StockInfo", conn)
    
    print('##### 更新技術分析資料 #####')
    
    # 更新資料庫
    for price_date in price_date_list :
        
        # API 使用次數 #
        user_count,api_request_limit = api_usage( token)
        if user_count > (api_request_limit - 100) :
            print('抓取資料速度過快（user_count＝ {} ，api_request_limit ＝ {}），等三十分鐘後再行抓取'.format(user_count,api_request_limit))
            time.sleep(30*60)
        
        if update_daily_price is True :
            # 股價日成交資訊 TaiwanStockPrice：一次拿特定日期，所有資料(只限 backer、sponsor 使用) #
            sqlcmd = "SELECT * FROM DailyPrice WHERE Date='{}'".format(price_date)
            df     = pd.read_sql_query(sqlcmd, conn)
            if df.empty is True :
                while True :
                    try :
                        df = api.taiwan_stock_daily(start_date=price_date,)
                    except Exception as e:
                        print('日K：日期{}發生錯誤{}，重試'.format(price_date,e))
                        continue
                    break
                if df.empty is not True :
                    print('日K：{}'.format(price_date))
                    df_daily_price = df.drop(columns=['spread','Trading_turnover'])
                    df_daily_price = df_daily_price.rename(columns={'date':'Date','stock_id':'StockID','Trading_Volume':'Volume','Trading_money':'Value','open':'Open','max':'High','min':'Low','close':'Close'})
                    # 保存格式：日期、股票代碼、開盤價、最高價、最低價、收盤價、成交量與成交值
                    df_daily_price = df_daily_price[['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value']]
                    # 排除掉非TaiwanStockInfo內的股票
                    df_daily_price = df_daily_price[df_daily_price['StockID'].isin(df_stock_info['StockID'].to_list())]
                    df_daily_price.to_sql('DailyPrice', conn, if_exists='append', index=False)
                else :    
                    time.sleep(1)
            else :
                #print('日K：日期{}資料已存在於資料庫中'.format(price_date))
                time.sleep(1)
        
        if update_weekly_price is True :
            # 台股週 K 資料表 TaiwanStockWeekPrice (只限 backer、sponsor 會員使用) ： 一次拿特定日期，所有資料(只限 backer、sponsor 使用) #
            sqlcmd = "SELECT * FROM WeeklyPrice WHERE Date='{}'".format(price_date)
            df     = pd.read_sql_query(sqlcmd, conn)
            if df.empty is True :
                url = "https://api.finmindtrade.com/api/v4/data"
                parameter = {
                    "dataset": "TaiwanStockWeekPrice",
                    "start_date": price_date,
                    "token": token,
                }
                while True :
                    resp = requests.get(url, params=parameter)
                    if resp.status_code == 200 :
                        break
                    else :
                        print('週K：日期{}發生錯誤，回應狀態碼 ＝ {}'.format(price_date,resp.status_code))
                        if resp.status_code == 402 :
                            time.sleep(10*60)
                data = resp.json()
                df_weekly_price = pd.DataFrame(data["data"])
                if df_weekly_price.empty is not True :
                    print('週K：{}'.format(price_date))
                    df_weekly_price = df_weekly_price.drop(columns=['yweek','spread','trading_turnover'])
                    df_weekly_price = df_weekly_price.rename(columns={'date':'Date','stock_id':'StockID','trading_volume':'Volume','trading_money':'Value','open':'Open','max':'High','min':'Low','close':'Close'})
                    # 保存格式：日期、股票代碼、開盤價、最高價、最低價、收盤價、成交量與成交值
                    df_weekly_price = df_weekly_price[['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value']]
                    # 排除掉非TaiwanStockInfo內的股票
                    df_weekly_price = df_weekly_price[df_weekly_price['StockID'].isin(df_stock_info['StockID'].to_list())]
                    df_weekly_price.to_sql('WeeklyPrice', conn, if_exists='append', index=False)
                else :    
                    time.sleep(1)
            else :
                #print('週K：日期{}資料已存在於資料庫中'.format(price_date))
                time.sleep(1)
        
        if update_monthly_price is True :
            # 台股月 K 資料表 TaiwanStockMonthPrice (只限 backer、sponsor 會員使用) ： 一次拿特定日期，所有資料(只限 backer、sponsor 使用) #
            sqlcmd = "SELECT * FROM MonthlyPrice WHERE Date='{}'".format(price_date)
            df     = pd.read_sql_query(sqlcmd, conn)
            if df.empty is True :
                url = "https://api.finmindtrade.com/api/v4/data"
                parameter = {
                    "dataset": "TaiwanStockMonthPrice",
                    "start_date": price_date,
                    "token": token, 
                }
                while True :
                    resp = requests.get(url, params=parameter)
                    if resp.status_code == 200 :
                        break
                    else :
                        print('月K：日期{}發生錯誤，回應狀態碼 ＝ {}'.format(price_date,resp.status_code))
                        if resp.status_code == 402 :
                            time.sleep(10*60)
                data = resp.json()
                df_monthly_price = pd.DataFrame(data["data"])
                if df_monthly_price.empty is not True :
                    print('月K：{}'.format(price_date))
                    df_monthly_price = df_monthly_price.drop(columns=['ymonth','spread','trading_turnover'])
                    df_monthly_price = df_monthly_price.rename(columns={'date':'Date','stock_id':'StockID','trading_volume':'Volume','trading_money':'Value','open':'Open','max':'High','min':'Low','close':'Close'})
                    # 保存格式：日期、股票代碼、開盤價、最高價、最低價、收盤價、成交量與成交值
                    df_monthly_price = df_monthly_price[['Date', 'StockID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value']]
                    # 排除掉非TaiwanStockInfo內的股票
                    df_monthly_price = df_monthly_price[df_monthly_price['StockID'].isin(df_stock_info['StockID'].to_list())]
                    df_monthly_price.to_sql('MonthlyPrice', conn, if_exists='append', index=False)
                else :    
                    time.sleep(1)
            else :
                #print('月K：日期{}資料已存在於資料庫中'.format(price_date))
                time.sleep(1)

    # 修改資料庫
    conn.commit()
    
    # 關閉資料庫
    conn.close()