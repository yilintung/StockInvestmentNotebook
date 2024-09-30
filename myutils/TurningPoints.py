#
# @file TurningPoints.py
#

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

##### 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/rolling_window.py #####

# Checks if there is a local top detected at curr index
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    
    return top

# Checks if there is a local top detected at curr index
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    
    return bottom

def rw_extremes(data: np.array, order:int):
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = confirmation index
            # top[1] = index of top
            # top[2] = price of top
            top = [i, i - order, data[i - order]]
            tops.append(top)
        if rw_bottom(data, i, order):
            # bottom[0] = confirmation index
            # bottom[1] = index of bottom
            # bottom[2] = price of bottom
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)
    return tops, bottoms


##### 尋找轉折點函式 #####
# prices         : 價格 (DataFrame)
# mode           : 模式 (str)
# order          : 最小化過濾器的距離參數，最小峰值寬度ｍｉｎ＿ｗｉｄｔｈ＝ｏｒｄｅｒ＊２＋１ (int)
# smoothing      : 平滑化參數 （當模式為'close'時有效） (int)
# real_body      : 以實體為主，轉折點會因波峰或波谷的狀況而選擇地使用開盤價 （當模式為'close'時有效） (bool)
# rolling_window : 使用neurotrader的"Rolling Window"方式尋找轉折點 (bool)
def FindingTurningPoints(prices, mode = 'close', order = 1, smoothing = 1,real_body = False,rolling_window = False) :
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
    if mode == 'high_low' :
        prices_high=np.array(in_prices['High'])
        prices_low=np.array(in_prices['Low'])
    elif mode == 'close' :
        if smoothing > 1 :
            # 若為平滑化的收盤價,使用向前與向後傳播非空值的方法填充空(NaN)值;並將窗口標籤設置為窗口索引的中心，以正確反映趨勢
            smooth_close_prices = np.array(in_prices['Close'].rolling(window=smoothing,center=True).mean().fillna(method='bfill').fillna(method='ffill'))
            prices_high=smooth_close_prices
            prices_low=smooth_close_prices
        else :
            prices_close=np.array(in_prices['Close'])
            prices_high=prices_close
            prices_low=prices_close
    elif mode == 'open_close' :
        prices_oepn  = np.array(in_prices['Open'])
        prices_close = np.array(in_prices['Close'])
        prices_high_list = []
        prices_low_list  = []
        for idx in range(0,len(prices_oepn)) :
            if prices_oepn[idx] >= prices_close[idx] :
                prices_high_list.append(prices_oepn[idx])
                prices_low_list.append(prices_close[idx])
            else :
                prices_high_list.append(prices_close[idx])
                prices_low_list.append(prices_oepn[idx])
        prices_high = np.array(prices_high_list)
        prices_low  = np.array(prices_low_list)
    else :
        return None
    if len(prices_high) != len(prices_low) :
        return None
    total_price = len(prices_high)
    # 找出轉折點，並保存其索引
    if rolling_window is True :
        _,local_min = rw_extremes(prices_low,order)
        local_max,_ = rw_extremes(prices_high,order)
        local_min = np.array(local_min)
        local_max = np.array(local_max)
        local_min_idx = local_min[:,1]
        local_max_idx = local_max[:,1]
    else :
        local_min_idx = argrelextrema(prices_low,np.less,order=order)[0]
        local_max_idx = argrelextrema(prices_high,np.greater,order=order)[0]
        local_min_idx = np.array(local_min_idx)
        local_max_idx = np.array(local_max_idx)
    # 合併局部最大(波峰)與局部最小(波谷)轉折點
    local_max_min_point=[]
    point_type = ''
    for idx in range(total_price):
        if idx in local_max_idx :
            if point_type != 'HI' :
                # 波谷轉波峰
                point_type = 'HI'
                if mode == 'high_low' :
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['High'],point_type))
                elif mode == 'close' :
                    real_body_price = in_prices.iloc[idx]['Close']
                    if real_body is True and in_prices.iloc[idx]['Open'] > in_prices.iloc[idx]['Close'] :
                        real_body_price = in_prices.iloc[idx]['Open']
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
                elif mode == 'open_close' :
                    real_body_price = in_prices.iloc[idx]['Close']
                    if in_prices.iloc[idx]['Open'] >= in_prices.iloc[idx]['Close'] :
                        real_body_price = in_prices.iloc[idx]['Open']
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
            else :
                # 同為波峰，選最高者
                previous_price = local_max_min_point[-1][2]
                if mode == 'high_low' :
                    current_price = in_prices.iloc[idx]['High']
                    if current_price > previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 新增目前這一筆
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['High'],point_type))
                elif mode == 'close' :
                    current_price = in_prices.iloc[idx]['Close']
                    if current_price > previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 新增目前這一筆
                        real_body_price = in_prices.iloc[idx]['Close']
                        if real_body is True and in_prices.iloc[idx]['Open'] > in_prices.iloc[idx]['Close'] :
                            real_body_price = in_prices.iloc[idx]['Open']
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
                elif mode == 'open_close' :
                    current_price = in_prices.iloc[idx]['Close']
                    if in_prices.iloc[idx]['Open'] >= in_prices.iloc[idx]['Close'] :
                        current_price = in_prices.iloc[idx]['Open']
                    if current_price > previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 新增目前這一筆
                        real_body_price = in_prices.iloc[idx]['Close']
                        if in_prices.iloc[idx]['Open'] >= in_prices.iloc[idx]['Close'] :
                            real_body_price = in_prices.iloc[idx]['Open']
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
        elif idx in local_min_idx :
            if point_type != 'LO' :
                # 波峰轉波谷
                point_type = 'LO'
                if mode == 'high_low' :
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['Low'],point_type))
                elif mode == 'close' :
                    real_body_price = in_prices.iloc[idx]['Close']
                    if real_body is True and in_prices.iloc[idx]['Open'] < in_prices.iloc[idx]['Close'] :
                        real_body_price = in_prices.iloc[idx]['Open']
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
                elif mode == 'open_close' :
                    real_body_price = in_prices.iloc[idx]['Close']
                    if in_prices.iloc[idx]['Open'] < in_prices.iloc[idx]['Close'] :
                        real_body_price = in_prices.iloc[idx]['Open']
                    local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
            else :
                # 同為波谷，選最低者
                previous_price = local_max_min_point[-1][2]
                if mode == 'high_low' :
                    current_price = in_prices.iloc[idx]['Low']
                    if current_price < previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 改新增目前這一筆
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],in_prices.iloc[idx]['Low'],point_type))
                elif mode == 'close' :
                    current_price = in_prices.iloc[idx]['Close']
                    if current_price < previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 改新增目前這一筆
                        real_body_price = in_prices.iloc[idx]['Close']
                        if real_body is True and in_prices.iloc[idx]['Open'] < in_prices.iloc[idx]['Close'] :
                            real_body_price = in_prices.iloc[idx]['Open']
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
                elif mode == 'open_close' :
                    current_price = in_prices.iloc[idx]['Close']
                    if in_prices.iloc[idx]['Open'] < in_prices.iloc[idx]['Close'] :
                        current_price = in_prices.iloc[idx]['Open']
                    if current_price < previous_price :
                        # 移除前一筆
                        previous_point = local_max_min_point.pop()
                        # 改新增目前這一筆
                        real_body_price = in_prices.iloc[idx]['Close']
                        if in_prices.iloc[idx]['Open'] < in_prices.iloc[idx]['Close'] :
                            real_body_price = in_prices.iloc[idx]['Open']
                        local_max_min_point.append((idx,in_prices.iloc[idx]['Date'],real_body_price,point_type))
    local_max_min_point = pd.DataFrame(local_max_min_point,columns=['DateIndex','Date','Price','Type'])
    # 轉換為輸出格式
    max_min = local_max_min_point.set_index('DateIndex')
    return local_min_idx,local_max_idx,max_min