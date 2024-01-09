#
# @file StockGPT.py
#

##### 來源 ： https://colab.research.google.com/drive/1gneFQWNI6PfkuE9WvN26bVMm702weH8D#scrollTo=j1x0glPsNJe2

from openai import OpenAI, OpenAIError  # 串接 OpenAI API
import yfinance as yf
import pandas as pd                     # 資料處理套件
import numpy as np
import datetime as dt                   # 時間套件
import requests
from bs4 import BeautifulSoup

import os
from dotenv import load_dotenv, find_dotenv

class StockGPT :

    ##### StockGPT類別的建構子 #####
    def __init__(self) :
        load_dotenv(find_dotenv())
        api_key = os.environ.get('OPENAI_API_KEY_TOKEN')
        self.client = OpenAI(api_key = api_key) # 建立 OpenAI 物件
        self.name_df = self.stock_name()
        self.get_stock_price_progress = False
        self.news_use_stock_name = False

    # 從 yfinance 取得一周股價資料
    def stock_price(self, stock_id="大盤", days = 10):
        if stock_id == "大盤":
            stock_id="^TWII"
        else:
            stock_id += ".TW"

        end = dt.date.today() # 資料結束時間
        start = end - dt.timedelta(days=days) # 資料開始時間
        # 下載資料
        df = yf.download(stock_id, start=start, progress=self.get_stock_price_progress)

        # 更換列名
        df.columns = ['開盤價', '最高價', '最低價',
                      '收盤價', '調整後收盤價', '成交量']

        data = {
            '日期': df.index.strftime('%Y-%m-%d').tolist(),
            '收盤價': df['收盤價'].tolist(),
            '每日報酬': df['收盤價'].pct_change().tolist(),
            '漲跌價差': df['調整後收盤價'].diff().tolist()
        }

        return data
    
    # 基本面資料
    def stock_fundamental(self, stock_id= "大盤"):
        if stock_id == "大盤":
            return None

        stock_id += ".TW"
        stock = yf.Ticker(stock_id)

        # 營收成長率
        quarterly_revenue_growth = np.round(stock.quarterly_financials.loc["Total Revenue"].pct_change(-1).dropna().tolist(), 2)

        # 每季EPS
        quarterly_eps = np.round(stock.quarterly_financials.loc["Basic EPS"].dropna().tolist(), 2)

        # EPS季增率
        quarterly_eps_growth = np.round(stock.quarterly_financials.loc["Basic EPS"].pct_change(-1).dropna().tolist(), 2)

        # 轉換日期
        dates = [date.strftime('%Y-%m-%d') for date in stock.quarterly_financials.columns]

        data = {
            '季日期': dates[:len(quarterly_revenue_growth)],  # 以最短的數據列表長度為准，確保數據對齊
            '營收成長率': quarterly_revenue_growth.tolist(),
            'EPS': quarterly_eps.tolist(),
            'EPS 季增率': quarterly_eps_growth.tolist()
        }

        return data

    # 新聞資料
    def stock_news(self, stock_name ="大盤"):
        if stock_name == "大盤":
            stock_name="台股 -盤中速報"

        data=[]
        # 取得 Json 格式資料
        json_data = requests.get(f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={stock_name}&limit=5&page=1').json()

        # 依照格式擷取資料
        items=json_data['data']['items']
        for item in items:
            # 網址、標題和日期
            news_id = item["newsId"]
            title = item["title"]
            publish_at = item["publishAt"]
            # 使用 UTC 時間格式
            utc_time = dt.datetime.utcfromtimestamp(publish_at)
            formatted_date = utc_time.strftime('%Y-%m-%d')
            # 前往網址擷取內容
            url = requests.get(f'https://news.cnyes.com/'
                               f'news/id/{news_id}').content
            soup = BeautifulSoup(url, 'html.parser')
            p_elements=soup .find_all('p')
            # 提取段落内容
            p=''
            for paragraph in p_elements[4:]:
                p+=paragraph.get_text()
            data.append([stock_name, formatted_date ,title,p])
        return data
    
    # 取得全部股票的股號、股名
    def stock_name(self):
      #print("線上讀取股號、股名、及產業別")

      response = requests.get('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2')
      url_data = BeautifulSoup(response.text, 'html.parser')
      stock_company = url_data.find_all('tr')

      # 資料處理
      data = [
          (row.find_all('td')[0].text.split('\u3000')[0].strip(),
           row.find_all('td')[0].text.split('\u3000')[1],
           row.find_all('td')[4].text.strip())
          for row in stock_company[2:] if len(row.find_all('td')[0].text.split('\u3000')[0].strip()) == 4
      ]

      df = pd.DataFrame(data, columns=['股號', '股名', '產業別'])

      return df
      
    # 取得股票名稱
    def get_stock_name(self, stock_id):
        return self.name_df.set_index('股號').loc[stock_id, '股名']
        
    # 建立 GPT 3.5-16k 模型
    def get_reply(self, messages):
        try:
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo-1106",
                messages = messages
            )
            reply = response.choices[0].message.content
        except OpenAIError as err:
            reply = f"發生 {err.type} 錯誤\n{err.message}"
        return reply

    # 建立訊息指令(Prompt)
    def generate_content_msg(self, stock_id):

        stock_name = self.get_stock_name(stock_id) if stock_id != "大盤" else stock_id

        price_data = self.stock_price(stock_id)
        if self.news_use_stock_name is True :
            news_data = self.stock_news(stock_name)
        else :
            news_data = self.stock_news(stock_id)

        content_msg = f'請依據以下資料來進行分析並給出一份完整的分析報告:\n'

        content_msg += f'近期價格資訊:\n {price_data}\n'

        if stock_id != "大盤":
            stock_value_data = self.stock_fundamental(stock_id)
            content_msg += f'每季營收資訊：\n {stock_value_data}\n'

        content_msg += f'近期新聞資訊: \n {news_data}\n'
        content_msg += f'請給我{stock_name}近期的趨勢報告,請以詳細、\
            嚴謹及專業的角度撰寫此報告,並提及重要的數字, reply in 繁體中文'

        return content_msg

    # StockGPT
    def stock_gpt(self, stock_id):
        content_msg = self.generate_content_msg(stock_id)

        msg = [{
            "role": "system",
            "content": f"你現在是一位專業的證券分析師, 你會統整近期的股價\
            、基本面、新聞資訊等方面並進行分析, 然後生成一份專業的趨勢分析報告"
        }, {
            "role": "user",
            "content": content_msg
        }]

        reply_data = self.get_reply(msg)

        return reply_data