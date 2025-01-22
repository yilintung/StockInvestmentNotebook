import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import mplfinance as mpf
import numpy as np
from scipy.signal import argrelextrema

##### 偵測轉折點函式 #####
# prices    : 價格 (DataFrame)
# mode      : 模式 (str) ： 有'close'、'high_low'與'open_close'三種，預設為'close'
# period    : 時間窗週期 (int)
# smoothing : 平滑化參數 （當模式為'close'時有效） (int)
def DetectTurningPoints(prices, mode = 'open_close', period = 1, smoothing = 1) :
    # 價格資料確認與處理
    if prices is None or type(prices) is not pd.core.frame.DataFrame :
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
    prices_date  = np.array(in_prices['Date'])
    prices_close = np.array(in_prices['Close'])
    if mode == 'close' :
        if smoothing > 1 :
            # 若為平滑化的收盤價,使用向前與向後傳播非空值的方法填充空(NaN)值;並將窗口標籤設置為窗口索引的中心，以正確反映趨勢
            prices_smooth_close = np.array(in_prices['Close'].rolling(window=smoothing,center=True).mean().fillna(method='bfill').fillna(method='ffill'))
            prices_high = prices_smooth_close
            prices_low  = prices_smooth_close
            # 將模式改為'smooth_close'
            mode = 'smooth_close'
        else:
            prices_high = prices_close
            prices_low  = prices_close
    elif mode == 'high_low' :
        prices_high = np.array(in_prices['High'])
        prices_low  = np.array(in_prices['Low'])
    elif mode == 'open_close' :
        open_prices  = np.array(in_prices['Open'])
        close_prices = prices_close
        if len(open_prices) != len(close_prices) :
            return None
        prices_high_list = []
        prices_low_list  = []
        for idx in range(0,len(open_prices)) :
            if open_prices[idx] >= close_prices[idx] :
                prices_high_list.append(open_prices[idx])
                prices_low_list.append(close_prices[idx])
            else :
                prices_high_list.append(close_prices[idx])
                prices_low_list.append(open_prices[idx])
        prices_high = np.array(prices_high_list)
        prices_low  = np.array(prices_low_list)
    else :
        return None
    if len(prices_high) != len(prices_low) or len(prices_high) != len(prices_close) or len(prices_high) != len(prices_date) :
        return None
    total_price = len(prices_high)
    # 時間窗週期轉換為峰谷【「峰到峰」或「谷到谷」】最小寬度（ｍｉｎ＿ｗｉｄｔｈ）
    min_width = period if (period % 2) else (period + 1)
    # 峰谷最小寬度（ｍｉｎ＿ｗｉｄｔｈ）轉換為最小化過濾器的距離參數（ｏｒｄｅｒ）
    order = (min_width - 1) // 2
    order = 1 if order < 1 else order
    ### DEBUG ###
    print('ＤＥＢＵＧ ： 峰谷最小寬度 ＝ {} ， 最小化過濾器的距離參數 ＝ {} '.format(min_width,order))
    # 找出轉折點，並保存其索引
    local_min_idx = argrelextrema(prices_low,np.less,order=order)[0]
    local_max_idx = argrelextrema(prices_high,np.greater,order=order)[0]
    local_min_idx = np.array(local_min_idx)
    local_max_idx = np.array(local_max_idx)
    # 合併局部最大(波峰)與局部最小(波谷)轉折點
    local_max_min_point = []
    point_type = ''
    for idx in range(total_price):
        if idx in local_max_idx :
            if point_type != 'HI' :
                # 波谷轉波峰
                point_type = 'HI'
                if mode == 'smooth_close' :
                    local_max_min_point.append((idx,prices_date[idx],prices_close[idx],point_type))
                else :
                    local_max_min_point.append((idx,prices_date[idx],prices_high[idx],point_type))
            else :
                # 同為波峰，選最高者
                previous_price = local_max_min_point[-1][2]
                current_price  = prices_high[idx]
                if current_price > previous_price :
                    # 移除前一筆
                    previous_point = local_max_min_point.pop()
                    # 新增目前這一筆
                    if mode == 'smooth_close' :
                        local_max_min_point.append((idx,prices_date[idx],prices_close[idx],point_type))
                    else :
                        local_max_min_point.append((idx,prices_date[idx],prices_high[idx],point_type))
                else :
                    pass
        elif idx in local_min_idx :
            if point_type != 'LO' :
                # 波峰轉波谷
                point_type = 'LO'
                if mode == 'smooth_close' :
                    local_max_min_point.append((idx,prices_date[idx],prices_close[idx],point_type))
                else :
                    local_max_min_point.append((idx,prices_date[idx],prices_low[idx],point_type))
            else :
                # 同為波谷，選最低者
                previous_price = local_max_min_point[-1][2]
                current_price  = prices_low[idx]
                if current_price < previous_price :
                    # 移除前一筆
                    previous_point = local_max_min_point.pop()
                    # 新增目前這一筆
                    if mode == 'smooth_close' :
                        local_max_min_point.append((idx,prices_date[idx],prices_close[idx],point_type))
                    else :
                        local_max_min_point.append((idx,prices_date[idx],prices_low[idx],point_type))
                else :
                    pass
    local_max_min_point = pd.DataFrame(local_max_min_point,columns=['DateIndex','Date','Price','Type'])
    # 轉換為輸出格式
    max_min = local_max_min_point.set_index('DateIndex')
    return local_min_idx,local_max_idx,max_min


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err;

def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])

def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)

##### 擬合趨勢線函式 #####
# prices         : 價格 (DataFrame)
# mode           : 模式 (str)
def TrendlineAutomation(prices, mode = 'close') :
    # 價格資料確認與處理
    if prices is None or type(prices) is not pd.core.frame.DataFrame :
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
    prices_close=np.array(in_prices['Close'])
    if mode == 'close' :
        pass
    elif mode == 'high_low' :
        prices_high = np.array(in_prices['High'])
        prices_low = np.array(in_prices['Low'])
    elif mode == 'open_close' :
        open_prices  = np.array(in_prices['Open'])
        close_prices = prices_close
        prices_high_list = []
        prices_low_list  = []
        for idx in range(0,len(open_prices)) :
            if open_prices[idx] >= close_prices[idx] :
                prices_high_list.append(open_prices[idx])
                prices_low_list.append(close_prices[idx])
            else :
                prices_high_list.append(close_prices[idx])
                prices_low_list.append(open_prices[idx])
        prices_high = np.array(prices_high_list)
        prices_low  = np.array(prices_low_list)
    else :
        return None
    # 擬合趨勢線
    if mode == 'close' :
        trendlines = fit_trendlines_single(prices_close)
    elif mode == 'high_low' or mode == 'open_close' :
        trendlines = fit_trendlines_high_low(prices_high,prices_low,prices_close)
    # 輸出格式處理
    trendline_start_index = in_prices.iloc[0].name
    trendline_end_index = in_prices.iloc[-1].name
    trendline_start_date = in_prices.iloc[0]['Date'].strftime('%Y-%m-%d')
    trendline_end_date = in_prices.iloc[-1]['Date'].strftime('%Y-%m-%d')
    support_slope  = trendlines[0][0]
    support_intercept = trendlines[0][1]
    support_trendline_start_price = support_slope * trendline_start_index + support_intercept
    support_trendline_end_price = support_slope * trendline_end_index + support_intercept
    resist_slope = trendlines[1][0]
    resist_intercept = trendlines[1][1]
    resist_trendline_start_price = resist_slope * trendline_start_index + resist_intercept
    resist_trendline_end_price = resist_slope * trendline_end_index + resist_intercept
    # 回傳資料處理
    ret_trendlines =  [[(trendline_start_date,support_trendline_start_price),(trendline_end_date,support_trendline_end_price)],
                       [(trendline_start_date,resist_trendline_start_price),(trendline_end_date,resist_trendline_end_price)]]
    return (trendlines,ret_trendlines)


# 單元測試函式：偵測轉折點
def test_DetectTurningPoints(prices, sma, mode, period = 1, smoothing = 1) :
    _,_,max_min = DetectTurningPoints(prices,mode,period,smoothing)
    print("轉折點總筆數 ＝ {:d}".format(max_min.shape[0]))
    if max_min.shape[0] < 1 :
        return None
    display(max_min)

    # 設定K線格式
    mc = mpf.make_marketcolors(up='xkcd:light red', down='xkcd:almost black', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    # 轉折點
    turning_points_len = len(np.array(prices['Close']))
    turning_points = np.array([np.nan]*turning_points_len)
    for point in max_min.iterrows() :
        turning_points[point[0]] = point[1]['Price']
    # 轉折線
    turning_point_lines = []
    for idx in range(0,len(max_min)) :
        if (idx + 1) == len(max_min) :
            break
        this_point = max_min.iloc[idx]
        next_point = max_min.iloc[idx+1]
        turning_point_lines.append([[this_point['Date'],this_point['Price']],[next_point['Date'],next_point['Price']]])
        
    # 設定均線與轉折點
    apds = [
        mpf.make_addplot(turning_points,type='scatter',marker='o',markersize=25,color='xkcd:sky blue'),
        mpf.make_addplot(sma['SMA5'],width=1.0,color='xkcd:maroon'),
        mpf.make_addplot(sma['SMA10'],width=1.0,color='xkcd:cyan'),
        mpf.make_addplot(sma['SMA20'],width=1.5,color='xkcd:violet'),
        mpf.make_addplot(sma['SMA60'],width=0.5,color='xkcd:dark orange')
    ]
    
    # 繪出K線圖
    kwargs = dict(type='candle', style=s, figratio=(19,10), addplot=apds, alines=dict(alines=turning_point_lines, linewidths=0.8, colors='xkcd:orange yellow', alpha=0.4), datetime_format='%Y-%m-%d')
    mpf.plot(prices,**kwargs)
    return max_min

# 單元測試函式：趨勢線自動化
def test_TrendlineAutomation(prices, mode, range_start_date, range_end_date, resistance = True, support = True) :
    # 區間內畫出趨勢線
    params,lines = TrendlineAutomation(prices[range_start_date:range_end_date],mode)
    if resistance is False and support is True :
        lines = lines[0]
        colors = 'xkcd:red'
    elif resistance is True and support is False :
        lines = lines[1]
        colors = 'xkcd:blue'
    else :
        colors = ('xkcd:red','xkcd:blue')
    # 設定K線格式
    mc = mpf.make_marketcolors(up='xkcd:light red', down='xkcd:almost black', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    # 設定支撐線與壓力線
    seq_of_seq_of_points=lines
    # 繪出K線圖
    kwargs = dict(type='candle', style=s, figratio=(19,10), alines=dict(alines=seq_of_seq_of_points, linewidths=1.0, colors=colors, alpha=0.6), datetime_format='%Y-%m-%d')
    mpf.plot(prices,**kwargs)
    return params,lines