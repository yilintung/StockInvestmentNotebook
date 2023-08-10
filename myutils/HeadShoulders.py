#
# @file HeadShoulders.py
#

import pandas as pd
import numpy as np
from typing import List
from collections import deque
from dataclasses import dataclass
from myutils.TurningPoints import rw_top, rw_bottom

##### 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/head_shoulders.py #####

@dataclass
class HSPattern:

    # True if inverted, False if not. Inverted is "bullish" according to technical analysis dogma
    inverted: bool

    # Indices of the parts of the H&S pattern
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
   
    # Price of the parts of the H&S pattern. _p stands for price.
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
   
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1

    neck_start: float = -1
    neck_end: float = -1

    # Attributes
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

def compute_pattern_r2(data: np.array, pat: HSPattern):

    line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
    line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
    
    line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
    line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
    
    line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - pat.l_armpit)
    line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
    
    line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
    line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
    
    line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
    line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
    
    line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
    line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
    
    raw_data = data[pat.start_i:pat.break_i]
    hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
    mean = np.mean(raw_data)

    ss_res = np.sum( (raw_data - hs_model) ** 2.0 )
    ss_tot = np.sum( (raw_data - mean) ** 2.0 )

    r2 = 1.0 - ss_res / ss_tot
    return r2


def check_hs_pattern(extrema_indices: List[int], data: np.array, i:int, early_find: bool = False) -> HSPattern:
    ''' Returns a HSPattern if found, or None if not found ''' 
    # Unpack list
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1

    # Head must be higher than shoulders
    if data[head] <= max(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are above the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint  or data[r_shoulder] < l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None
        
    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope

    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] > r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] > neck_val:
            return None

    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] < neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=False)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val

    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    pat.pattern_r2 = compute_pattern_r2(data, pat)

    # I experiemented with r-squared as a filter for H&S, but this can delay recognition.
    # It didn't seem terribly potent, may be useful as a filter in conjunction with other attributes
    # if one wanted to add a machine learning layer before trading these patterns. 

    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat


def check_ihs_pattern(extrema_indices: List[int], data: np.array, i:int, early_find: bool = False) -> HSPattern:
    
    # Unpack list
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]
    
    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit+1: i].argmin() + 1

    # Head must be lower than shoulders
    if data[head] >= min(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are below the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint  or data[r_shoulder] > l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None

    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    
    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] < r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] < neck_val:
            return None
   
    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=True)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat


def find_hs_patterns(data: np.array, order:int, early_find:bool = False):
    assert(order >= 1)
    
    # head and shoulders top checked from/after a confirmed bottom (before right shoulder)
    # head and shoulders bottom checked from/after a confirmed top 
    
    last_is_top = False
    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5) # -1 for bottoms 1 for tops

    # Lock variables to prevent finding the same pattern multiple times
    hs_lock = False
    ihs_lock = False

    ihs_patterns = [] # Inverted (bullish)
    hs_patterns = []  # Regular (bearish)
    for i in range(len(data)):

        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock = False
            last_is_top = True
        
        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock = False
            last_is_top = False

        if len(recent_extrema) < 5:
            continue
        
        hs_alternating = True
        ihs_alternating = True
        
        if last_is_top:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            ihs_extrema = list(recent_extrema)[1:5]
            hs_extrema = list(recent_extrema)[0:4]
        else:
            
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            ihs_extrema = list(recent_extrema)[0:4]
            hs_extrema = list(recent_extrema)[1:5]
        
        if ihs_lock or not ihs_alternating:
            ihs_pat = None
        else:
            ihs_pat = check_ihs_pattern(ihs_extrema, data, i, early_find)

        if hs_lock or not hs_alternating:
            hs_pat = None
        else:
            hs_pat = check_hs_pattern(hs_extrema, data, i, early_find)

        if hs_pat is not None:
            hs_lock = True
            hs_patterns.append(hs_pat)
        
        if ihs_pat is not None:
            ihs_lock = True
            ihs_patterns.append(ihs_pat)

    return hs_patterns, ihs_patterns


##### 尋找頭肩型態函式 #####
# prices         : 價格 (DataFrame)
# order          : 最小化過濾器的距離參數，最小峰值寬度ｍｉｎ＿ｗｉｄｔｈ＝ｏｒｄｅｒ＊２＋１ (int)
# early_find     : 為True時會尋找還沒有完成(突破頸線)的型態
def FindingHeadShoulderPatterns(prices, order = 1, early_find = False) :
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
    prices_close=np.array(in_prices['Close'])
    # 尋找頭肩型態
    hs_patterns, ihs_patterns = find_hs_patterns(prices_close, order=order, early_find=early_find)
    # 轉換為輸出格式
    patterns = []
    for pattern in hs_patterns:
        pattern_points = []
        # 頭肩型態的左肩
        pattern_points.append([pattern.l_shoulder,in_prices.iloc[pattern.l_shoulder]['Date'].strftime("%Y-%m-%d"),pattern.l_shoulder_p])
        # 頭肩型態的左腋窩
        pattern_points.append([pattern.l_armpit,in_prices.iloc[pattern.l_armpit]['Date'].strftime("%Y-%m-%d"),pattern.l_armpit_p])
        # 頭肩型態的右肩
        pattern_points.append([pattern.r_shoulder,in_prices.iloc[pattern.r_shoulder]['Date'].strftime("%Y-%m-%d"),pattern.r_shoulder_p])
        # 頭肩型態的右腋窩
        pattern_points.append([pattern.r_armpit,in_prices.iloc[pattern.r_armpit]['Date'].strftime("%Y-%m-%d"),pattern.r_armpit_p])
        # 頭肩型態的頭
        pattern_points.append([pattern.head,in_prices.iloc[pattern.head]['Date'].strftime("%Y-%m-%d"),pattern.head_p])
        
        # 頸線開始與結束日期及價格
        neckline_start_date  = in_prices.iloc[pattern.start_i]['Date'].strftime("%Y-%m-%d")
        neckline_start_price = pattern.neck_start
        neckline_end_date    = in_prices.iloc[pattern.break_i]['Date'].strftime("%Y-%m-%d")
        neckline_end_price   = pattern.neck_end
        neckline_info = {'start_date' : neckline_start_date,'start_price' : neckline_start_price, 'end_date' : neckline_end_date, 'end_price' : neckline_end_price}
        
        # 頭肩型態的頭部資訊
        head_info = { 'width': pattern.head_width, 'height': pattern.head_height }
        
        patterns.append({ 'type': 'Top', 'points': pattern_points, 'neckline': neckline_info, 'head': head_info })
    for pattern in ihs_patterns:
        pattern_points = []
        # 頭肩型態的左肩
        pattern_points.append([pattern.l_shoulder,in_prices.iloc[pattern.l_shoulder]['Date'].strftime("%Y-%m-%d"),pattern.l_shoulder_p])
        # 頭肩型態的左腋窩
        pattern_points.append([pattern.l_armpit,in_prices.iloc[pattern.l_armpit]['Date'].strftime("%Y-%m-%d"),pattern.l_armpit_p])
        # 頭肩型態的右肩
        pattern_points.append([pattern.r_shoulder,in_prices.iloc[pattern.r_shoulder]['Date'].strftime("%Y-%m-%d"),pattern.r_shoulder_p])
        # 頭肩型態的右腋窩
        pattern_points.append([pattern.r_armpit,in_prices.iloc[pattern.r_armpit]['Date'].strftime("%Y-%m-%d"),pattern.r_armpit_p])
        # 頭肩型態的頭
        pattern_points.append([pattern.head,in_prices.iloc[pattern.head]['Date'].strftime("%Y-%m-%d"),pattern.head_p])
        
        # 頸線開始與結束日期及價格
        neckline_start_date  = in_prices.iloc[pattern.start_i]['Date'].strftime("%Y-%m-%d")
        neckline_start_price = pattern.neck_start
        neckline_end_date    = in_prices.iloc[pattern.break_i]['Date'].strftime("%Y-%m-%d")
        neckline_end_price   = pattern.neck_end
        neckline_info = {'start_date' : neckline_start_date,'start_price' : neckline_start_price, 'end_date' : neckline_end_date, 'end_price' : neckline_end_price}
        
        # 頭肩型態的頭部資訊
        head_info = { 'width': pattern.head_width, 'height': pattern.head_height }
        
        # 頭肩底
        patterns.append({ 'type': 'Bottom', 'points': pattern_points, 'neckline': neckline_info, 'head': head_info })
    return patterns