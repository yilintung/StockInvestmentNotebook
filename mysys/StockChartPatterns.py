import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from collections import defaultdict

##### 【內部函式】 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/trendline_automation.py #####

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
    
##### 【內部函式】 正規化 ##### 

def normalization(data) :
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
##### 股票技術型態識別 #####
class StockChartPatterns :

    ### StockChartPatterns類別的建構子 ###
    def __init__( self, prices, debug = False) :
        # 設定除錯旗標
        self._debug = debug
        # 價格資料(DataFrame)確認與處理
        if prices is None or type(prices) is not pd.core.frame.DataFrame :
            raise ValueError
        self._prices = prices.copy()
        if 'Open' not in self._prices.columns or 'High' not in self._prices.columns or 'Low' not in self._prices.columns or 'Close' not in self._prices.columns or 'Volume' not in self._prices.columns :
            raise ValueError
        if 'Date' not in self._prices.columns and self._prices.index.dtype == 'datetime64[ns]' :
            self._prices.index.name = 'Date'
            self._prices = self._prices.reset_index()
        if 'Date' not in self._prices.columns :
            raise ValueError
    
    ### 列印除錯訊息之內部方法 ###
    def _debug_print( self, msg) :
        if self._debug is True :
            print("ＤＥＢＵＧ ： {}".format(msg))
    
    ### 偵測轉折點方法 ###
    def DetectTurningPoints( self, mode = 'close', order = 1, smoothing = 1) :
        # 模式確認與處理
        prices_date  = np.array(self._prices['Date'])
        prices_close = np.array(self._prices['Close'])
        if mode == 'close' :
            if smoothing > 1 :
                # 若為平滑化的收盤價,使用向前與向後傳播非空值的方法填充空(NaN)值;並將窗口標籤設置為窗口索引的中心，以正確反映趨勢
                # prices_smooth_close = np.array(self._prices['Close'].rolling(window=smoothing,center=True).mean().fillna(method='bfill').fillna(method='ffill'))
                prices_smooth_close = np.array(self._prices['Close'].rolling(window=smoothing,center=True).mean().bfill().ffill())
                prices_high = prices_smooth_close
                prices_low  = prices_smooth_close
                # 將模式改為'smooth_close'
                mode = 'smooth_close'
            else:
                prices_high = prices_close
                prices_low  = prices_close
        elif mode == 'high_low' :
            prices_high = np.array(self._prices['High'])
            prices_low  = np.array(self._prices['Low'])
        elif mode == 'open_close' :
            prices_open  = np.array(self._prices['Open'])
            if len(prices_open) != len(prices_close) :
                self._debug_print('不可能發生的錯誤 ： len(prices_open) != len(prices_close)')
                return None
            prices_high_list = []
            prices_low_list  = []
            for idx in range(0,len(prices_open)) :
                if prices_open[idx] >= prices_close[idx] :
                    prices_high_list.append(prices_open[idx])
                    prices_low_list.append(prices_close[idx])
                else :
                    prices_high_list.append(prices_close[idx])
                    prices_low_list.append(prices_open[idx])
            prices_high = np.array(prices_high_list)
            prices_low  = np.array(prices_low_list)
        else :
            return None
        if len(prices_high) != len(prices_low) or len(prices_high) != len(prices_close) or len(prices_high) != len(prices_date) :
            self._debug_print('不可能發生的錯誤 ： len(prices_high) != len(prices_low) or len(prices_high) != len(prices_close) or len(prices_high) != len(prices_date)')
            return None
        total_price = len(prices_high)
        # 最小化過濾器的距離參數（ｏｒｄｅｒ）與最小峰值／谷值寬度（ｍｉｎ＿ｗｉｄｔｈ）間之關係 ： ｍｉｎ＿ｗｉｄｔｈ＝ｏｒｄｅｒ＊２＋１
        order = 1 if order < 1 else order
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
                    if mode == 'smooth_close' :
                        current_price  = prices_close[idx]
                    else :
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
                    if mode == 'smooth_close' :
                        current_price  = prices_close[idx]
                    else :
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
        
    ### 擬合趨勢線內部方法 ###
    def _fit_trendlines( self, prices, mode = 'close') :
        # 模式確認與處理
        prices_close=np.array(prices['Close'])
        if mode == 'close' :
            pass
        elif mode == 'high_low' :
            prices_high = np.array(prices['High'])
            prices_low  = np.array(prices['Low'])
        elif mode == 'open_close' :
            prices_open      = np.array(prices['Open'])
            prices_high_list = []
            prices_low_list  = []
            for idx in range(0,len(prices_open)) :
                if prices_open[idx] >= prices_close[idx] :
                    prices_high_list.append(prices_open[idx])
                    prices_low_list.append(prices_close[idx])
                else :
                    prices_high_list.append(prices_close[idx])
                    prices_low_list.append(prices_open[idx])
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
        trendline_start_index         = 0
        trendline_end_index           = (prices.iloc[-1].name) - (prices.iloc[0].name)
        self._debug_print('趨勢線開始索引 ＝ {}（價格資料開始索引 ＝ {}），趨勢線結束索引 ＝ {}（價格資料結束索引 ＝ {}）'.format(trendline_start_index,prices.iloc[0].name,trendline_end_index,prices.iloc[-1].name))
        trendline_start_date          = prices.iloc[0]['Date'].strftime('%Y-%m-%d')
        trendline_end_date            = prices.iloc[-1]['Date'].strftime('%Y-%m-%d')
        support_slope                 = trendlines[0][0]
        support_intercept             = trendlines[0][1]
        support_trendline_start_price = support_slope * trendline_start_index + support_intercept
        support_trendline_end_price   = support_slope * trendline_end_index + support_intercept
        resist_slope                  = trendlines[1][0]
        resist_intercept              = trendlines[1][1]
        resist_trendline_start_price  = resist_slope * trendline_start_index + resist_intercept
        resist_trendline_end_price    = resist_slope * trendline_end_index + resist_intercept
        # 回傳資料處理
        ret_trendlines = [[(trendline_start_date,support_trendline_start_price),(trendline_end_date,support_trendline_end_price)],
                          [(trendline_start_date,resist_trendline_start_price),(trendline_end_date,resist_trendline_end_price)]]
        return (trendlines,ret_trendlines)
    
    ### 自動趨勢線方法 ###
    def TrendlineAutomation( self, mode = 'close') :
        return self._fit_trendlines( self._prices, mode)
    
    ### 型態識別方法 ###
    # （１）這個方法是使用轉折點方式進行型態識別，並參考TradingView圖表形態( https://tw.tradingview.com/support/solutions/43000706927/ )內容進行開發
    # （２）參考Algorithmic Chart Pattern Detection( https://analyzingalpha.com/algorithmic-chart-pattern-detection )文章之演算法
    # （３）未驗證：三角旗形
    def Recognition( self, max_bars = 120, turning_point_args = {'mode':'open_close', 'order':5}) :
        # 儲存型態實例
        patterns = defaultdict(list)
        
        # 轉折點引數確認
        turning_point_args_list = []
        if type(turning_point_args) is dict :
            turning_point_args_list.append(turning_point_args)
        elif type(turning_point_args) is list :
            turning_point_args_list = turning_point_args
        elif type(turning_point_args) is tuple :
            turning_point_args_list = list(turning_point_args)
        else :
            return patterns
        self._debug_print('轉折點引數列表 ＝ {}'.format(turning_point_args_list))
        
        # 各種引數之轉折點列表
        turning_point_info_list = []
        # 依據引數偵測轉折點
        for argument in turning_point_args_list :
            if type(argument) is dict and 'mode' in argument and 'order' in argument :
                if argument['mode'] == 'close' and 'smoothing' in argument :
                    _,_,turning_points = self.DetectTurningPoints( mode=argument['mode'], order=argument['order'], smoothing=argument['smoothing'])
                else :
                    _,_,turning_points = self.DetectTurningPoints( mode=argument['mode'], order=argument['order'])
                turning_point_info_list.append((argument,turning_points))
        
        for turning_point_info in turning_point_info_list :
            
            # 元組解包
            argument,turning_points = turning_point_info
            
            ## 四個轉折點之型態 ##
            
            #循環迭代轉折點數據
            for i in range(4, (len(turning_points)+1)):
                
                # 在變數window中一次儲存4個局部最小值和局部最大值點
                window = turning_points.iloc[i-4:i]
                
                # 型態必須在max_bars內發揮出來（預設為 120）
                if window.index[-1] - window.index[0] > max_bars:
                    self._debug_print('四個轉折點區間超過 max_bars ＝ {} 範圍'.format(max_bars))
                    continue
        
                # 儲存4個轉折點以檢查條件
                e1      = window.iloc[0]['Price']
                e2      = window.iloc[1]['Price']
                e3      = window.iloc[2]['Price']
                e4      = window.iloc[3]['Price']
                rtop_g1 = np.mean([e1,e3])
                rtop_g2 = np.mean([e2,e4])
        
                if e2 > e1 and e4 > e3 and e3 > e1 and e4 > e2 and (e2 - e1) > (e4 - e3) and abs(((e4 - e3) - (e2 - e1)) / (e2 - e1)) > 0.25 :
                    patterns['上升楔形'].append({'argument':argument,'window':window})
                elif e1 > e2 and e3 > e4 and e1 > e3 and e2 > e4 and (e1 - e2) > (e3 - e4) and abs(((e3 - e4) - (e1 - e2)) / (e1 - e2)) > 0.25 :
                    patterns['下降楔形'].append({'argument':argument,'window':window})
                elif (e1 > e2) and (abs(e1-rtop_g1)/rtop_g1 < 0.0075) and (abs(e3-rtop_g1)/rtop_g1 < 0.0075) and \
                    (abs(e2-rtop_g2)/rtop_g2 < 0.0075) and (abs(e4-rtop_g2)/rtop_g2 < 0.0075) and (min(e1, e3) > max(e2, e4)):
                    patterns['矩形'].append({'argument':argument,'window':window})
                elif e1 < e3 and e3 < e2 and e3 < e4 and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e2-e4) <= np.mean([e2,e4])*0.05):
                    patterns['雙重頂'].append({'argument':argument,'window':window})
                elif e1 > e3 and e3 > e2 and e3 > e4 and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e2-e4) <= np.mean([e2,e4])*0.05):
                    patterns['雙重底'].append({'argument':argument,'window':window})
            
            ## 五個轉折點之型態 ##
             
            #循環迭代轉折點數據
            for i in range(5, len(turning_points)+1):
        
                # 在變數window中一次儲存5個局部最小值和局部最大值點
                window = turning_points.iloc[i-5:i]
        
                # 型態必須在max_bars內發揮出來（預設為 120）
                if window.index[-1] - window.index[0] > max_bars:
                    self._debug_print('五個轉折點區間超過 max_bars ＝ {} 範圍'.format(max_bars))
                    continue
        
                # 儲存5個轉折點以檢查條件
                e1      = window.iloc[0]['Price']
                e2      = window.iloc[1]['Price']
                e3      = window.iloc[2]['Price']
                e4      = window.iloc[3]['Price']
                e5      = window.iloc[4]['Price']
        
                if (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
                    patterns['三角形'].append({'argument':argument,'window':window})
                elif (e1 > e2) and (e1 > e3) and (e1 > e5) and (e2 < e3) and (e4 < e5) and (e2 < e4) and (e3 < e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.10) and \
                    (abs(e2-e4) >= np.mean([e2,e4])*0.015) and \
                    (abs(e3-e5) >= np.mean([e3,e5])*0.015):
                    # 上升旗形(看跌)
                    patterns['上升旗形'].append({'argument':argument,'window':window})
                elif (e1 < e2) and (e1 < e3) and (e1 < e5) and (e2 > e3) and (e4 > e5) and (e2 > e4) and (e3 > e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.10) and \
                    (abs(e2-e4) >= np.mean([e2,e4])*0.015) and \
                    (abs(e3-e5) >= np.mean([e3,e5])*0.015):
                    # 下降旗形(看漲)
                    patterns['下降旗形'].append({'argument':argument,'window':window})
                elif (e1 > e2) and (e1 > e3) and (e1 > e5) and (e2 < e3) and (e4 < e5) and (e2 <= e4) and (e3 > e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.10) and \
                    (abs(e3-e5) >= np.mean([e3,e5])*0.015):
                    # 未驗證
                    patterns['看跌三角旗形'].append({'argument':argument,'window':window})
                elif (e1 < e2) and (e1 < e3) and (e1 < e5) and (e2 > e3) and (e4 > e5) and (e2 > e4) and (e3 <= e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.10) and \
                    (abs(e2-e4) >= np.mean([e2,e4])*0.015) :
                    # 未驗證
                    patterns['看漲三角旗形'].append({'argument':argument,'window':window})
            
            ## 六個轉折點之型態 ##
            #循環迭代轉折點數據
            for i in range(6, len(turning_points)+1):
        
                # 在變數window中一次儲存6個局部最小值和局部最大值點
                window = turning_points.iloc[i-6:i]
        
                # 型態必須在max_bars內發揮出來（預設為 120）
                if window.index[-1] - window.index[0] > max_bars:
                    self._debug_print('六個轉折點區間超過 max_bars ＝ {} 範圍'.format(max_bars))
                    continue
        
                # 儲存6個轉折點以檢查條件
                e1      = window.iloc[0]['Price']
                e2      = window.iloc[1]['Price']
                e3      = window.iloc[2]['Price']
                e4      = window.iloc[3]['Price']
                e5      = window.iloc[4]['Price']
                e6      = window.iloc[5]['Price']

                if (e3 < e1) and (e5 < e1) and (e2 < e3) and (e4 < e2) and (e4 < e6) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e3-e5) <= np.mean([e3,e5])*0.05) and \
                    (abs(e2-e4) >= np.mean([e2,e6])*0.05) and \
                    (abs(e4-e6) >= np.mean([e2,e6])*0.05):
                    patterns['頭肩底'].append({'argument':argument,'window':window})
                elif (e3 > e1) and (e5 > e1) and (e2 > e3) and (e4 > e2) and (e4 > e6) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e3-e5) <= np.mean([e3,e5])*0.05) and \
                    (abs(e2-e4) >= np.mean([e2,e6])*0.05) and \
                    (abs(e4-e6) >= np.mean([e2,e6])*0.05):
                    patterns['頭肩頂'].append({'argument':argument,'window':window})
                elif (e3 < e1) and (e5 < e1) and (e2 < e3) and (e4 < e5) and (e6 < e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e3-e5) <= np.mean([e3,e5])*0.05) and \
                    (abs(e2-e4) <= np.mean([e2,e4,e6])*0.10) and \
                    (abs(e4-e6) <= np.mean([e2,e4,e6])*0.10) :
                    patterns['三重底'].append({'argument':argument,'window':window})
                elif (e3 > e1) and (e5 > e1) and (e2 > e3) and (e4 > e5) and (e6 > e5) and \
                    (abs(e1-e3) >= np.mean([e1,e3])*0.03) and \
                    (abs(e3-e5) <= np.mean([e3,e5])*0.05) and \
                    (abs(e2-e4) <= np.mean([e2,e4,e6])*0.10) and \
                    (abs(e4-e6) <= np.mean([e2,e4,e6])*0.10) :
                    patterns['三重頂'].append({'argument':argument,'window':window})

        return patterns
        
    ### 底部型態識別方法 ###
    # （１）「底型（底部型態）反轉」交易策略專用之型態識別方法
    # （２）底部型態需滿足突破後往上空間至少要有１５％之條件且目標價不能過前高
    def RecognitionBottom( self ,turning_point_args = {'mode':'open_close', 'order':5}, valid_pattern_check = True) :
        # 儲存型態實例
        patterns = defaultdict(list)
        
        # 轉折點引數確認
        turning_point_args_list = []
        if type(turning_point_args) is dict :
            turning_point_args_list.append(turning_point_args)
        elif type(turning_point_args) is list :
            turning_point_args_list = turning_point_args
        elif type(turning_point_args) is tuple :
            turning_point_args_list = list(turning_point_args)
        else :
            return patterns
        self._debug_print('轉折點引數列表 ＝ {}'.format(turning_point_args_list))
        
        # 各種引數之轉折點列表
        turning_point_info_list = []
        # 依據引數偵測轉折點
        for argument in turning_point_args_list :
            if type(argument) is dict and 'mode' in argument and 'order' in argument :
                if argument['mode'] == 'close' and 'smoothing' in argument :
                    _,_,turning_points = self.DetectTurningPoints( mode=argument['mode'], order=argument['order'], smoothing=argument['smoothing'])
                else :
                    _,_,turning_points = self.DetectTurningPoints( mode=argument['mode'], order=argument['order'])
                turning_point_info_list.append((argument,turning_points))
        
        # 底部型態列表
        bottom_pattern_list = []
        
        for turning_point_info in turning_point_info_list :
            
            # 元組解包
            argument,turning_points = turning_point_info
            turning_points_len = len(turning_points)
            
            # 偵測底部型態
            turning_points_chk_start_idx = 0
            check_next_pattern           = False
            while True :
                check_next_pattern            = False
                previous_high_point           = None
                bottom_pattern_high_point     = None
                bottom_pattern_low_point      = None
                bottom_pattern_start_point    = None
                bottom_pattern_end_date       = None
                bottom_pattern_breakout_index = None
                bottom_pattern_breakout_date  = None
                bottom_pattern_breakout_price = None
                turning_points_chk_range      = range(turning_points_chk_start_idx,turning_points_len)
                
                self._debug_print('轉折點資料確認範圍 （開始索引 ＝ {} ，結束索引 ＝ {}） '.format(turning_points_chk_start_idx,turning_points_len-1))
                
                # 轉折高點確認
                prev_hi_point = None
                for idx in turning_points_chk_range:
                    point = turning_points.iloc[idx]
                    if point['Type'] == 'HI' :
                        if prev_hi_point is not None :
                            if previous_high_point is None or bottom_pattern_high_point is None :
                                if ((prev_hi_point['Price'] - point['Price']) / point['Price']) > 0.15 :
                                    # 設定底部型態之前高點
                                    previous_high_point       = prev_hi_point
                                    # 設定底部型態之最高點
                                    bottom_pattern_high_point = point
                                else :
                                    # 此轉折高點與先前轉折高點間不足１５％空間
                                    pass
                            else :
                                if point['Price'] > previous_high_point['Price'] :
                                    # 結束底部型態確認：此轉折高點超過底部型態之前高點
                                    turning_points_chk_range     = range(turning_points_chk_start_idx,idx)
                                    turning_points_chk_start_idx = idx
                                    check_next_pattern           = True
                                    self._debug_print('此轉折高點超過底部型態之前高點，結束底部型態確認。此轉折點 ＝ \n{}'.format(point))
                                    break
                                if point['Price'] > bottom_pattern_high_point['Price'] :
                                    # 此轉折點高於底部型態之最高點
                                    if ((previous_high_point['Price'] - point['Price']) / point['Price']) > 0.15 :
                                        # 此轉折高點與底部型態之前高點間仍有１５％空間
                                        if ((point['Price'] - prev_hi_point['Price']) / prev_hi_point['Price']) > 0.15 :
                                            # 結束底部型態確認：此轉折高點超過先前轉折高點１５％
                                            turning_points_chk_range     = range(turning_points_chk_start_idx,idx)
                                            turning_points_chk_start_idx = idx
                                            check_next_pattern           = True
                                            self._debug_print('此轉折高點超過先前轉折高點１５％，結束底部型態確認。此轉折點 ＝ \n{}'.format(point))
                                            break
                                        else :
                                            # 更新底部型態之最高點
                                            bottom_pattern_high_point = point
                                    else :
                                        # 結束底部型態確認：此轉折高點與底部型態之前高點間不足１５％空間
                                        turning_points_chk_range     = range(turning_points_chk_start_idx,idx)
                                        turning_points_chk_start_idx = idx
                                        check_next_pattern           = True
                                        self._debug_print('此轉折高點與底部型態之前高點間不足１５％空間，結束底部型態確認。此轉折點 ＝ \n{}'.format(point))
                                        break
                                if bottom_pattern_start_point is None and point['Price'] < bottom_pattern_high_point['Price'] and ((bottom_pattern_high_point['Price'] - point['Price']) / point['Price']) > 0.10:
                                    # 前一高點超過目前高點１０％，進行特別處理
                                    # TODO : 程序待進一步確認
                                    self._debug_print('＃ 前轉折高點超過此轉折高點１０％，進行特別處理。前一轉折高點 ＝ \n{}\n此轉折高點 ＝ \n{}'.format(bottom_pattern_high_point,point))
                                    previous_high_point        = bottom_pattern_high_point
                                    bottom_pattern_high_point  = point
                        # 設定或更新先前轉折高點
                        prev_hi_point = point
                
                # 轉折低點確認
                for idx in turning_points_chk_range:
                    point = turning_points.iloc[idx]
                    if point['Type'] == 'LO' :
                        if bottom_pattern_start_point is None and idx > 0 and previous_high_point is not None and turning_points.iloc[idx-1]['Date'] == previous_high_point['Date'] :
                            # 設定底部型態開始之轉折點
                            bottom_pattern_start_point = point
                        if bottom_pattern_start_point is not None :
                            if bottom_pattern_low_point is None :
                                # 設定底部型態最低點
                                bottom_pattern_low_point  = point
                            elif point['Price'] < bottom_pattern_low_point['Price'] :
                                # 更新底部型態最低點
                                bottom_pattern_low_point = point
                    # 保存底部型態最後個轉折點資料，之後用於計算底部型態結束索引與日期
                    bottom_pattern_last_point = point
                
                # 確定底部型態並設定頸線
                if previous_high_point is not None and bottom_pattern_high_point is not None and bottom_pattern_start_point is not None and bottom_pattern_low_point is not None :
                    # 設定底部型態開始與結束之價格索引與日期
                    bottom_pattern_start_idx   = bottom_pattern_start_point.name
                    bottom_pattern_start_date  = bottom_pattern_start_point['Date'].strftime("%Y-%m-%d")
                    bottom_pattern_end_idx     = bottom_pattern_last_point.name
                    bottom_pattern_end_date    = bottom_pattern_last_point['Date'].strftime("%Y-%m-%d")
                    # 調整底部型態結束索引與日期
                    botton_end_check_start_idx = bottom_pattern_end_idx
                    botton_end_check_end_idx   = self._prices.shape[0]
                    if check_next_pattern is True :
                        botton_end_check_end_idx = turning_points.iloc[turning_points_chk_start_idx].name
                    for idx in range(botton_end_check_start_idx,botton_end_check_end_idx) :
                        if self._prices.iloc[idx]['Close'] > bottom_pattern_high_point['Price'] or self._prices.iloc[idx]['Open'] > bottom_pattern_high_point['Price'] :
                            bottom_pattern_end_idx  = idx - 1
                            bottom_pattern_end_date = self._prices.iloc[bottom_pattern_end_idx]['Date'].strftime("%Y-%m-%d")
                            break
                    self._debug_print('底部型態開始索引 ＝ {} （日期 ＝ {} ） ， 底部型態結束索引 ＝ {} （日期 ＝ {} ）'.format(bottom_pattern_start_idx,bottom_pattern_start_date,bottom_pattern_end_idx,bottom_pattern_end_date))
                    # 設定趨勢線區間價格資料
                    range_prices = self._prices[bottom_pattern_start_idx:(bottom_pattern_end_idx+1)]
                    # 使用趨勢線做為頸線，這邊擬合趨勢線的參數mode = 'open_close'
                    params,lines = self._fit_trendlines(range_prices,'open_close')
                    # 取得頸線(line2：壓力線)相關參數
                    neckline_slope     = params[1][0]
                    neckline_intercept = params[1][1]
                    # 將趨勢線進行正規化，求頸線真正的角度
                    line_prices_list = []
                    line_prices_list.append(lines[0][0][1])
                    line_prices_list.append(lines[0][1][1])
                    line_prices_list.append(lines[1][0][1])
                    line_prices_list.append(lines[1][1][1])
                    line_prices      = np.array(line_prices_list)
                    line_prices_norm = normalization(line_prices)
                    line1_x          = [0.0, 1.0]
                    line1_y          = [line_prices_norm[0],line_prices_norm[1]]
                    line2_x          = [0.0, 1.0]
                    line2_y          = [line_prices_norm[2],line_prices_norm[3]]
                    line1_angle      = np.rad2deg(np.arctan2(line1_y[1] - line1_y[0] , line1_x[1] - line1_x[0]))
                    line2_angle      = np.rad2deg(np.arctan2(line2_y[1] - line2_y[0] , line2_x[1] - line2_x[0]))
                    # 如果頸線正負角度過大，則以底部型態之最高點來設定直線頸線
                    neckline_angle   = abs(line2_angle)
                    # self._debug_print('頸線角度（絕對值） ＝ {:.2f}°'.format(neckline_angle))
                    # TODO : 判斷條件待確認？
                    if (line2_angle > 0 and neckline_angle > 5.0) or (line2_angle < 0 and neckline_angle > 15.0) :
                        neckline_slope     = 0.0
                        neckline_intercept = bottom_pattern_high_point['Price']
                        self._debug_print('頸線角度（絕對值）{:.2f}°過大，改用直線頸線（斜率 ＝ {:.2f}，截距 ＝ {:.2f}）'.format(neckline_angle,neckline_slope,neckline_intercept))
                    # 設定頸線開始位置與價格
                    previous_high_idx  = previous_high_point.name
                    neckline_start_idx = bottom_pattern_start_idx
                    for idx in range(previous_high_idx,bottom_pattern_start_idx) :
                        neckline_idx_price = (neckline_slope * (idx - bottom_pattern_start_idx)) + neckline_intercept
                        if self._prices.iloc[idx]['Close'] < neckline_idx_price or self._prices.iloc[idx]['Open'] < neckline_idx_price : 
                            neckline_start_idx = idx - 1
                            break
                    neckline_start_date  = self._prices.iloc[neckline_start_idx]['Date'].strftime("%Y-%m-%d")
                    neckline_start_price = (neckline_slope * (neckline_start_idx - bottom_pattern_start_idx)) + neckline_intercept
                    # 檢查底部型態結束後20根K線是否突破，若是則設定突破位置與價格
                    check_breakout_end_idx = (bottom_pattern_end_idx + 20) if (bottom_pattern_end_idx + 20) < self._prices.shape[0] else self._prices.shape[0]
                    self._debug_print('底部型態突破確認範圍 ： {} ～ {}'.format(bottom_pattern_end_idx,check_breakout_end_idx))
                    for idx in range(bottom_pattern_end_idx,check_breakout_end_idx) :
                        breakout_chk_price = (neckline_slope * (idx - bottom_pattern_start_idx)) + neckline_intercept
                        if ((self._prices.iloc[idx]['Close'] - breakout_chk_price) / breakout_chk_price) > 0.03 :
                            bottom_pattern_breakout_index = idx
                            bottom_pattern_breakout_date  = self._prices.iloc[idx]['Date'].strftime("%Y-%m-%d")
                            bottom_pattern_breakout_price = breakout_chk_price
                            self._debug_print("確認底部型態突破，收盤價超過頸線３％。價格索引 ＝ {} ， 日期 ＝ {} ， 突破時頸線價格 ＝ {:.2f}元 ".format(bottom_pattern_breakout_index,bottom_pattern_breakout_date,bottom_pattern_breakout_price))
                            break
                    # 設定頸線結束位置與價格
                    if check_next_pattern is True :
                        if bottom_pattern_breakout_index is not None :
                            neckline_end_idx = bottom_pattern_breakout_index
                        else :
                            neckline_end_idx = bottom_pattern_end_idx
                    else :
                        neckline_end_idx = self._prices.iloc[-1].name
                    neckline_end_date  = self._prices.iloc[neckline_end_idx]['Date'].strftime("%Y-%m-%d")
                    neckline_end_price = (neckline_slope * (neckline_end_idx - bottom_pattern_start_idx)) + neckline_intercept
                    
                    # 估算目標價
                    bottom_price_idx  = bottom_pattern_low_point.name
                    bottom_price_date = bottom_pattern_low_point['Date'].strftime("%Y-%m-%d")
                    bottom_price      = bottom_pattern_low_point['Price']
                    on_neckline_price = neckline_slope * (bottom_price_idx - bottom_pattern_start_idx) + neckline_intercept
                    if bottom_pattern_breakout_price is not None :
                        target_price = (on_neckline_price - bottom_price) + bottom_pattern_breakout_price
                    else :
                        target_price = (on_neckline_price - bottom_price) + neckline_end_price
                    self._debug_print('底部最低價格 ＝ {:.2f}元 ； 估算目標價 ＝ {:.2f}元'.format(bottom_price,target_price))
                    # 建立底部型態資訊
                    if bottom_pattern_breakout_date is not None and bottom_pattern_breakout_price is not None :
                        bottom_pattern = {
                            'previous_high_date':previous_high_point['Date'].strftime("%Y-%m-%d"),'previous_high_price':previous_high_point['Price'],
                            'bottom_pattern_start_date':bottom_pattern_start_date,'bottom_pattern_end_date':bottom_pattern_end_date,
                            'neckline_slope':neckline_slope,'neckline_intercept':neckline_intercept,
                            'neckline_start_date':neckline_start_date,'neckline_start_price':neckline_start_price,
                            'neckline_end_date':neckline_end_date,'neckline_end_price':neckline_end_price,
                            'bottom_pattern_breakout_date':bottom_pattern_breakout_date,'bottom_pattern_breakout_price':bottom_pattern_breakout_price,
                            'bottom_price_date':bottom_price_date,'bottom_price':bottom_price,
                            'on_neckline_price':on_neckline_price,'target_price':target_price
                                         }
                    else :
                        bottom_pattern = {
                            'previous_high_date':previous_high_point['Date'].strftime("%Y-%m-%d"),'previous_high_price':previous_high_point['Price'],
                            'bottom_pattern_start_date':bottom_pattern_start_date,'bottom_pattern_end_date':bottom_pattern_end_date,
                            'neckline_slope':neckline_slope,'neckline_intercept':neckline_intercept,
                            'neckline_start_date':neckline_start_date,'neckline_start_price':neckline_start_price,
                            'neckline_end_date':neckline_end_date,'neckline_end_price':neckline_end_price,
                            'bottom_price_date':bottom_price_date,'bottom_price':bottom_price,
                            'on_neckline_price':on_neckline_price,'target_price':target_price
                                         }
                    valid_bottom_pattern = True
                    if valid_pattern_check is True :
                        # 檢查是否為有效底部型態：(1)目標價不能過前高、(2)底部型態中有太過突出的高低點（代表可能有關鍵轉折點未被偵測）
                        if target_price >= previous_high_point['Price'] :
                            self._debug_print('目標價過前高，非有效底部型態 ： {}'.format(bottom_pattern))
                            valid_bottom_pattern = False
                        for idx in range(bottom_pattern_start_idx,bottom_pattern_end_idx) :
                            if ((self._prices.iloc[idx]['Close'] - bottom_pattern_high_point['Price']) / bottom_pattern_high_point['Price']) > 0.05 or \
                                ((self._prices.iloc[idx]['Open'] - bottom_pattern_high_point['Price']) / bottom_pattern_high_point['Price']) > 0.05 :
                                self._debug_print('底部型態中有突出５％的高點，非有效底部型態 ： {}'.format(bottom_pattern))
                                valid_bottom_pattern = False
                                break
                            if ((bottom_pattern_low_point['Price'] - self._prices.iloc[idx]['Close']) / self._prices.iloc[idx]['Close']) > 0.05 or \
                                ((bottom_pattern_low_point['Price'] - self._prices.iloc[idx]['Open']) / self._prices.iloc[idx]['Open']) > 0.05 :
                                self._debug_print('底部型態中有突出５％的低點，非有效底部型態 ： {}'.format(bottom_pattern))
                                valid_bottom_pattern = False
                                break
                    if valid_bottom_pattern is True :
                        # 將底部型態保存至列表中
                        bottom_pattern_list.append({'argument':argument,'bottom_pattern':bottom_pattern})
                
                if check_next_pattern is False :
                    break
        
        return bottom_pattern_list