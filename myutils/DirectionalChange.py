#
# @file DirectionalChange.py
#

import pandas as pd
import numpy as np

##### 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/directional_change.py #####

def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    
    up_zig = True # Last extreme is a bottom. Next is a top. 
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig: # Last extreme is a bottom
            if high[i] > tmp_max:
                # New high, update 
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma: 
                # Price retraced by sigma %. Top confirmed, record it
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # Setup for next bottom
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else: # Last extreme is a top
            if low[i] < tmp_min:
                # New low, update 
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma: 
                # Price retraced by sigma %. Bottom confirmed, record it
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms
    
    
##### 尋找方向變化點(或 ZigZag)函式 #####
# prices     : 價格 (DataFrame)
# percentage : 百分比 (int)
# mode       : 模式 (str)
def FindingDirectionalChangePoints( prices, percentage=5, mode='default') :
    # 價格資料確認與處理
    if prices is None and type(prices) is not DataFrame:
        return None
    in_prices = prices.copy()
    if 'Open' not in in_prices.columns or 'High' not in in_prices.columns or 'Low' not in in_prices.columns or 'Close' not in in_prices.columns :
        return None    
    if 'Date' not in in_prices.columns and in_prices.index.dtype == 'datetime64[ns]' :
        in_prices.index.name = 'Date'
        in_prices = in_prices.reset_index()
    if 'Date' not in in_prices.columns :
        return None
    # 模式確認與處理
    if mode == 'default' :
        prices_close=np.array(in_prices['Close'])
        prices_high=np.array(in_prices['High'])
        prices_low=np.array(in_prices['Low'])
    elif mode == 'close' :
        prices_close=np.array(in_prices['Close'])
        prices_high=np.array(in_prices['Close'])
        prices_low=np.array(in_prices['Close'])
    # 參數轉換    
    sigma = float(percentage) / 100.0
    # 呼叫directional_change函式    
    tops, bottoms = directional_change( close=prices_close, high=prices_high, low=prices_low, sigma=sigma)
    local_min = np.array(bottoms)
    local_max = np.array(tops)
    local_min_idx = local_min[:,1]
    local_max_idx = local_max[:,1]
    # 合併局部最大(波峰)與局部最小(波谷)轉折點
    local_max_min_point=[]
    total_price = len(prices_close)
    for idx in range(total_price):
        if mode == 'default' :
            if idx in local_max_idx :
                local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['High'],'HI'))
            elif idx in local_min_idx :
                local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['Low'],'LO'))
        elif mode == 'close' :
            if idx in local_max_idx :
                local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['Close'],'HI'))
            elif idx in local_min_idx :
                local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['Close'],'LO'))
    local_max_min_point = pd.DataFrame(local_max_min_point,columns=['DateIndex','Date','Price','Type'])
    # 轉換為輸出格式
    max_min = local_max_min_point.set_index('DateIndex')
    return local_min_idx,local_max_idx,max_min
