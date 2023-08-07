#
# @file TrendlineAutomation.py
#

import pandas as pd
import numpy as np

##### 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/trendline_automation.py #####

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
def FitTrendlines(prices, mode = 'close') :
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
    prices_close=np.array(in_prices['Close'])
    if mode == 'close' :
        pass
    elif mode == 'high_low' :
        prices_high=np.array(in_prices['High'])
        prices_low=np.array(in_prices['Low'])
    else :
        return None
    # 擬合趨勢線
    if mode == 'close' :
        trendlines = fit_trendlines_single(prices_close)
    elif mode == 'high_low' :
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