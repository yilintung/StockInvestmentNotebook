import os
import pandas as pd
import numpy as np
import datetime
import sqlite3

from talib.abstract import *
from .StockChartPatterns import StockChartPatterns
from collections import defaultdict

import io
import base64
import matplotlib.pyplot as plt
import matplotlib
import mplfinance as mpf
import json
import openai
import re

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from PIL import Image

##### 【公用函式】 來源 ： https://www.cnblogs.com/Rosaany/p/15561918.html #####
def get_monday_to_sunday(today, weekly=0):
    last = weekly * 7
    today = datetime.datetime.strptime(str(today), "%Y-%m-%d")
    monday = datetime.datetime.strftime(today - datetime.timedelta(today.weekday() - last), "%Y-%m-%d")
    monday_ = datetime.datetime.strptime(monday, "%Y-%m-%d")
    sunday = datetime.datetime.strftime(monday_ + datetime.timedelta(monday_.weekday() + 6), "%Y-%m-%d")
    return monday, sunday

##### 【公用函式】 日期轉索引 #####
def date_to_index(df_in,date_in) :
    number_array = df_in.index == date_in
    idx = 0
    for number_index in number_array :
        if number_index == True :
            break
        idx = idx + 1
    return idx
    
###### 【公用函式】 來源 ： https://stackoverflow.com/questions/73604477/i-am-trying-to-crop-an-image-to-remove-extra-space-in-python #####
def get_first_last(mask, axis: int):
    """ Find the first and last index of non-zero values along an axis in `mask` """
    mask_axis = np.argmax(mask, axis=axis) > 0
    a = np.argmax(mask_axis)
    b = len(mask_axis) - np.argmax(mask_axis[::-1])
    return int(a), int(b)

def crop_borders(img, crop_color):
    np_img = np.array(img)
    mask = (np_img != crop_color)[..., 0]   # compute a mask
    x0, x1 = get_first_last(mask, 0)        # find boundaries along x axis
    y0, y1 = get_first_last(mask, 1)        # find boundaries along y axis
    return img.crop((x0, y0, x1, y1)) 

##### 【內部函式】 型態識別除錯訊息輸出 #####
def pattern_recognition_debug_print(msg,debug=False) :
    if debug is True :
        print("ＤＥＢＵＧ ： {}".format(msg))

##### 【內部函式】 基於轉折點的型態識別之後處理程序 #####
def pattern_post_processing( prices , pattern_name, pattern_window ,debug = False) :
    
    # 回傳值
    pattern_return_dict = None
    
    if pattern_name == '雙重頂' or pattern_name == '雙重底' or pattern_name == '三重頂' or pattern_name == '三重底' or pattern_name == '頭肩頂' or pattern_name == '頭肩底' :
        
        pattern_first_date_idx  = pattern_window.iloc[0].name
        pattern_start_date_idx  = pattern_window.iloc[1].name
        pattern_end_date_idx    = pattern_window.iloc[-1].name
        
        is_breakout             = False
        
        neckline_start_date     = None
        neckline_end_date       = None
        neckline_start_price    = None
        neckline_end_price      = None
        
        pattern_type_char       = ''
        if '頂' in pattern_name :
            pattern_type_char = '頂'
        elif '底' in pattern_name :
            pattern_type_char = '底'
        
        target_price            = None 
        
        if pattern_name == '雙重頂' or pattern_name == '雙重底':
            
            neckline_index  = [ 2]
            
            neckline_price          = pattern_window.iloc[neckline_index[0]]['Price']
            for idx in range(pattern_first_date_idx,pattern_start_date_idx) :
                if pattern_type_char == '頂' :
                    if prices.iloc[idx]['Close'] > neckline_price or prices.iloc[idx]['Open'] > neckline_price :
                        neckline_start_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_start_price = neckline_price
                        break
                    else :
                        pass
                elif pattern_type_char == '底' :
                    if prices.iloc[idx]['Close'] < neckline_price or prices.iloc[idx]['Open'] < neckline_price :
                        neckline_start_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_start_price = neckline_price
                        break
                    else :
                        pass
            for idx in range(pattern_end_date_idx,date_to_index(prices,prices.iloc[-1].name)) :
                if pattern_type_char == '頂' :
                    if prices.iloc[idx]['Close'] < neckline_price or prices.iloc[idx]['Open'] < neckline_price :
                        neckline_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_end_price = neckline_price
                        break
                    else :
                        pass
                elif pattern_type_char == '底' :
                    if prices.iloc[idx]['Close'] > neckline_price or prices.iloc[idx]['Open'] > neckline_price :
                        neckline_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_end_price = neckline_price
                        break
                    else :
                        pass
            if neckline_end_date is None or neckline_end_price is None :
                # 型態進行中
                neckline_end_date  = prices.iloc[-1].name.strftime("%Y-%m-%d")
                neckline_end_price = neckline_price
            else :
                # 型態已跌破／突破
                is_breakout = True
        elif pattern_name == '三重頂' or pattern_name == '三重底' or pattern_name == '頭肩頂' or pattern_name == '頭肩底' :
            
            neckline_index = [ 2, 4]
            
            neckline_x = [date_to_index(prices,pattern_window.iloc[neckline_index[0]]['Date']),date_to_index(prices,pattern_window.iloc[neckline_index[1]]['Date'])]
            neckline_y = [pattern_window.iloc[neckline_index[0]]['Price'],pattern_window.iloc[neckline_index[1]]['Price']]
            neckline_slope,neckline_intercept = np.polyfit(neckline_x,neckline_y,1)
                    
            for idx in range(pattern_first_date_idx,pattern_start_date_idx) :
                on_nickline_price = neckline_slope * idx + neckline_intercept
                if pattern_type_char == '頂' :
                    if prices.iloc[idx]['Close'] > on_nickline_price or prices.iloc[idx]['Open'] > on_nickline_price :
                        neckline_start_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_start_price = on_nickline_price
                        break
                    else :
                        pass
                elif pattern_type_char == '底' :
                    if prices.iloc[idx]['Close'] < on_nickline_price or prices.iloc[idx]['Open'] < on_nickline_price :
                        neckline_start_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_start_price = on_nickline_price
                        break
                    else :
                        pass
            for idx in range(pattern_end_date_idx,date_to_index(prices,prices.iloc[-1].name)) :
                on_nickline_price = neckline_slope * idx + neckline_intercept
                if pattern_type_char == '頂' :
                    if prices.iloc[idx]['Close'] < on_nickline_price or prices.iloc[idx]['Open'] < on_nickline_price :
                        neckline_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_end_price = on_nickline_price
                        break
                    else :
                        pass
                elif pattern_type_char == '底' :
                    if prices.iloc[idx]['Close'] > on_nickline_price or prices.iloc[idx]['Open'] > on_nickline_price :
                        neckline_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                        neckline_end_price = on_nickline_price
                        break
                    else :
                        pass
            if neckline_end_date is None or neckline_end_price is None :
                # 型態進行中
                neckline_end_date  = prices.iloc[-1].name.strftime("%Y-%m-%d")
                neckline_end_price = neckline_slope * date_to_index(prices,prices.iloc[-1].name) + neckline_intercept
            else :
                # 型態已跌破／突破
                is_breakout = True
        # 確認頸線是否存在
        if neckline_start_date is None or neckline_end_date is None or neckline_start_price is None or neckline_end_price is None :
            pattern_recognition_debug_print('  ╳ 找不到頸線？',debug=debug)
        else :
            # 計算目標價
            if pattern_name == '雙重頂' or pattern_name == '雙重底':
                top_bottom_index = [ 1, 3]
            else :
                top_bottom_index = [ 1, 3, 5]
            if pattern_type_char == '頂' :
                top_point = pattern_window.iloc[top_bottom_index[0]]
                for idx in range(1,len(top_bottom_index)) :
                    if pattern_window.iloc[top_bottom_index[idx]]['Price'] > top_point['Price'] :
                        top_point = pattern_window.iloc[top_bottom_index[idx]]
                if pattern_name == '雙重頂' :
                    target_price      = neckline_price - (top_point['Price'] - neckline_price)
                    on_nickline_price = neckline_price
                elif pattern_name == '三重頂' or pattern_name == '頭肩頂' :
                    on_nickline_price = neckline_slope * date_to_index(prices,top_point['Date']) + neckline_intercept
                    target_price      = neckline_end_price - (top_point['Price'] - on_nickline_price)
            elif pattern_type_char == '底' :
                bottom_point = pattern_window.iloc[top_bottom_index[0]]
                for idx in range(1,len(top_bottom_index)) :
                    if pattern_window.iloc[top_bottom_index[idx]]['Price'] < bottom_point['Price'] :
                        bottom_point = pattern_window.iloc[top_bottom_index[idx]]
                if pattern_name == '雙重底' :
                    target_price      = (neckline_price - bottom_point['Price']) + neckline_price
                    on_nickline_price = neckline_price
                elif pattern_name == '三重底' or pattern_name == '頭肩底' :
                    on_nickline_price = neckline_slope * date_to_index(prices,bottom_point['Date']) + neckline_intercept
                    target_price      = (on_nickline_price - bottom_point['Price']) + neckline_end_price
            # 建立型態資料
            if pattern_type_char == '頂' :
                pattern_data_dict   = {
                    'neckline_start_date':neckline_start_date,'neckline_end_date':neckline_end_date,
                    'neckline_start_price':neckline_start_price,'neckline_end_price':neckline_end_price,
                    'top_point_date' : top_point['Date'] , 'top_point_price' : top_point['Price'], 'on_nickline_price':on_nickline_price,
                    'target_price' : target_price,'is_breakout' : is_breakout,
                    'window':pattern_window
                                      }
            elif pattern_type_char == '底' :
                pattern_data_dict   = {
                    'neckline_start_date':neckline_start_date,'neckline_end_date':neckline_end_date,
                    'neckline_start_price':neckline_start_price,'neckline_end_price':neckline_end_price,
                    'bottom_point_date' : bottom_point['Date'] , 'bottom_point_price' : bottom_point['Price'], 'on_nickline_price':on_nickline_price,
                    'target_price' : target_price,'is_breakout' : is_breakout,
                    'window':pattern_window
                                      }
            pattern_return_dict = {'類型' : '反轉型態', '型態' : pattern_name, '資料' : pattern_data_dict}
    elif pattern_name == '三角形' : 
        
        pattern_end_date_idx        = pattern_window.iloc[-1].name
        
        resistance_line_index       = [0,4]
        support_line_idx            = [1,3]
        
        resistance_line_start_date  = pattern_window.iloc[0]['Date'].strftime("%Y-%m-%d")
        resistance_line_end_date    = pattern_window.iloc[-1]['Date'].strftime("%Y-%m-%d")
        resistance_line_x           = [date_to_index(prices,pattern_window.iloc[resistance_line_index[0]]['Date']),date_to_index(prices,pattern_window.iloc[resistance_line_index[1]]['Date'])]
        resistance_line_y           = [pattern_window.iloc[resistance_line_index[0]]['Price'],pattern_window.iloc[resistance_line_index[1]]['Price']]
        resistance_line_slope,resistance_line_intercept = np.polyfit(resistance_line_x,resistance_line_y,1)
        resistance_line_start_price = resistance_line_slope * date_to_index(prices,resistance_line_start_date) + resistance_line_intercept
        resistance_line_end_price   = resistance_line_slope * date_to_index(prices,resistance_line_end_date) + resistance_line_intercept
                
        support_line_start_date     = resistance_line_start_date
        support_line_end_date       = resistance_line_end_date
        support_line_x              = [date_to_index(prices,pattern_window.iloc[support_line_idx[0]]['Date']),date_to_index(prices,pattern_window.iloc[support_line_idx[1]]['Date'])]
        support_line_y              = [pattern_window.iloc[support_line_idx[0]]['Price'],pattern_window.iloc[support_line_idx[1]]['Price']]
        support_line_slope,support_line_intercept = np.polyfit(support_line_x,support_line_y,1)
        support_line_start_price    = support_line_slope * date_to_index(prices,support_line_start_date) + support_line_intercept
        support_line_end_price      = support_line_slope * date_to_index(prices,support_line_end_date) + support_line_intercept
        
        if support_line_end_price > resistance_line_end_price :
            pattern_recognition_debug_print('  ╳ 無效的三角形型態！',debug=debug)
        else :
            # 趨勢線向後延伸
            for idx in range(pattern_end_date_idx+1,date_to_index(prices,prices.iloc[-1].name)) :
                tmp_resistance_price      = resistance_line_slope * idx + resistance_line_intercept
                tmp_support_price         = support_line_slope * idx + support_line_intercept
                
                resistance_line_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
                resistance_line_end_price = tmp_resistance_price
                support_line_end_date     = resistance_line_end_date
                support_line_end_price    = tmp_support_price
                
                if prices.iloc[idx]['Close'] > tmp_resistance_price or prices.iloc[idx]['Open'] > tmp_resistance_price or \
                  prices.iloc[idx]['Close'] < tmp_support_price or prices.iloc[idx]['Open'] < tmp_support_price :
                    break
            # 建立型態資料
            pattern_data_dict = {'resistance_line_start_date' : resistance_line_start_date , 'resistance_line_end_date' : resistance_line_end_date , \
                                 'resistance_line_start_price' : resistance_line_start_price, 'resistance_line_end_price' : resistance_line_end_price, \
                                 'support_line_start_date' : support_line_start_date, 'support_line_end_date' : support_line_end_date, \
                                 'support_line_start_price' : support_line_start_price , 'support_line_end_price' : support_line_end_price, \
                                 'window':pattern_window}
            pattern_return_dict = {'類型' : '盤整型態', '型態' : pattern_name, '資料' : pattern_data_dict}
    elif pattern_name == '上升楔形' or pattern_name == '下降楔形' :
        
        pattern_end_date_idx        = pattern_window.iloc[-1].name
        
        if pattern_name == '上升楔形' :
            resistance_line_index   = [1,3]
            support_line_idx        = [0,2]
        else :
            resistance_line_index   = [0,2]
            support_line_idx        = [1,3]
        
        resistance_line_start_date  = pattern_window.iloc[0]['Date'].strftime("%Y-%m-%d")
        resistance_line_end_date    = pattern_window.iloc[-1]['Date'].strftime("%Y-%m-%d")
        resistance_line_x           = [date_to_index(prices,pattern_window.iloc[resistance_line_index[0]]['Date']),date_to_index(prices,pattern_window.iloc[resistance_line_index[1]]['Date'])]
        resistance_line_y           = [pattern_window.iloc[resistance_line_index[0]]['Price'],pattern_window.iloc[resistance_line_index[1]]['Price']]
        resistance_line_slope,resistance_line_intercept = np.polyfit(resistance_line_x,resistance_line_y,1)
        resistance_line_start_price = resistance_line_slope * date_to_index(prices,resistance_line_start_date) + resistance_line_intercept
        resistance_line_end_price   = resistance_line_slope * date_to_index(prices,resistance_line_end_date) + resistance_line_intercept
                
        support_line_start_date     = resistance_line_start_date
        support_line_end_date       = resistance_line_end_date
        support_line_x              = [date_to_index(prices,pattern_window.iloc[support_line_idx[0]]['Date']),date_to_index(prices,pattern_window.iloc[support_line_idx[1]]['Date'])]
        support_line_y              = [pattern_window.iloc[support_line_idx[0]]['Price'],pattern_window.iloc[support_line_idx[1]]['Price']]
        support_line_slope,support_line_intercept = np.polyfit(support_line_x,support_line_y,1)
        support_line_start_price    = support_line_slope * date_to_index(prices,support_line_start_date) + support_line_intercept
        support_line_end_price      = support_line_slope * date_to_index(prices,support_line_end_date) + support_line_intercept
        
        # 趨勢線向後延伸
        for idx in range(pattern_end_date_idx+1,date_to_index(prices,prices.iloc[-1].name)) :
            tmp_resistance_price      = resistance_line_slope * idx + resistance_line_intercept
            tmp_support_price         = support_line_slope * idx + support_line_intercept
            
            resistance_line_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
            resistance_line_end_price = tmp_resistance_price
            support_line_end_date     = resistance_line_end_date
            support_line_end_price    = tmp_support_price
            
            if prices.iloc[idx]['Close'] > tmp_resistance_price or prices.iloc[idx]['Open'] > tmp_resistance_price or \
              prices.iloc[idx]['Close'] < tmp_support_price or prices.iloc[idx]['Open'] < tmp_support_price :
                break
        # 建立型態資料
        pattern_data_dict = {'resistance_line_start_date' : resistance_line_start_date , 'resistance_line_end_date' : resistance_line_end_date , \
                             'resistance_line_start_price' : resistance_line_start_price, 'resistance_line_end_price' : resistance_line_end_price, \
                             'support_line_start_date' : support_line_start_date, 'support_line_end_date' : support_line_end_date, \
                             'support_line_start_price' : support_line_start_price , 'support_line_end_price' : support_line_end_price, \
                             'window':pattern_window}
        pattern_return_dict = {'類型' : '反轉型態', '型態' : pattern_name, '資料' : pattern_data_dict}
    elif pattern_name == '矩形' :
        
        pattern_end_date_idx        = pattern_window.iloc[-1].name
        
        rect_upper_edge_price =  max(pattern_window.iloc[0]['Price'],pattern_window.iloc[2]['Price']) * 1.0
        rect_lower_edge_price =  min(pattern_window.iloc[1]['Price'],pattern_window.iloc[3]['Price']) * 1.0

        resistance_line_start_date = pattern_window.iloc[0]['Date'].strftime("%Y-%m-%d")
        resistance_line_end_date   = pattern_window.iloc[-1]['Date'].strftime("%Y-%m-%d")
        resistance_line_price      = rect_upper_edge_price
        
        support_line_start_date    = resistance_line_start_date
        support_line_end_date      = resistance_line_end_date
        support_line_price         = rect_lower_edge_price
         
        # 趨勢線向後延伸
        for idx in range(pattern_end_date_idx+1,date_to_index(prices,prices.iloc[-1].name)) :
            resistance_line_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
            support_line_end_date     = resistance_line_end_date
            
            if prices.iloc[idx]['Close'] > resistance_line_price or prices.iloc[idx]['Open'] > resistance_line_price or \
              prices.iloc[idx]['Close'] < support_line_price or prices.iloc[idx]['Open'] < support_line_price :
                break
        # 建立型態資料
        pattern_data_dict = {'resistance_line_start_date' : resistance_line_start_date , 'resistance_line_end_date' : resistance_line_end_date , \
                             'resistance_line_price' : resistance_line_price, \
                             'support_line_start_date' : support_line_start_date, 'support_line_end_date' : support_line_end_date, \
                             'support_line_price' : support_line_price, \
                             'window':pattern_window}
        pattern_return_dict = {'類型' : '盤整型態', '型態' : pattern_name, '資料' : pattern_data_dict}
    elif pattern_name == '上升旗形' or pattern_name == '下降旗形' or pattern_name == '看跌三角旗形' or pattern_name == '看漲三角旗形'  : 
        
        pattern_end_date_idx        = pattern_window.iloc[-1].name
        
        if pattern_name == '上升旗形' or pattern_name == '看跌三角旗形':
            resistance_line_index   = [2,4]
            support_line_idx        = [1,3]
        else :
            resistance_line_index   = [1,3]
            support_line_idx        = [2,4]
        
        resistance_line_start_date  = pattern_window.iloc[1]['Date'].strftime("%Y-%m-%d")
        resistance_line_end_date    = pattern_window.iloc[-1]['Date'].strftime("%Y-%m-%d")
        resistance_line_x           = [date_to_index(prices,pattern_window.iloc[resistance_line_index[0]]['Date']),date_to_index(prices,pattern_window.iloc[resistance_line_index[1]]['Date'])]
        resistance_line_y           = [pattern_window.iloc[resistance_line_index[0]]['Price'],pattern_window.iloc[resistance_line_index[1]]['Price']]
        resistance_line_slope,resistance_line_intercept = np.polyfit(resistance_line_x,resistance_line_y,1)
        resistance_line_start_price = resistance_line_slope * date_to_index(prices,resistance_line_start_date) + resistance_line_intercept
        resistance_line_end_price   = resistance_line_slope * date_to_index(prices,resistance_line_end_date) + resistance_line_intercept
                
        support_line_start_date     = resistance_line_start_date
        support_line_end_date       = resistance_line_end_date
        support_line_x              = [date_to_index(prices,pattern_window.iloc[support_line_idx[0]]['Date']),date_to_index(prices,pattern_window.iloc[support_line_idx[1]]['Date'])]
        support_line_y              = [pattern_window.iloc[support_line_idx[0]]['Price'],pattern_window.iloc[support_line_idx[1]]['Price']]
        support_line_slope,support_line_intercept = np.polyfit(support_line_x,support_line_y,1)
        support_line_start_price    = support_line_slope * date_to_index(prices,support_line_start_date) + support_line_intercept
        support_line_end_price      = support_line_slope * date_to_index(prices,support_line_end_date) + support_line_intercept
        
        # 趨勢線向後延伸
        for idx in range(pattern_end_date_idx+1,date_to_index(prices,prices.iloc[-1].name)) :
            tmp_resistance_price      = resistance_line_slope * idx + resistance_line_intercept
            tmp_support_price         = support_line_slope * idx + support_line_intercept
            
            resistance_line_end_date  = prices.iloc[idx].name.strftime("%Y-%m-%d")
            resistance_line_end_price = tmp_resistance_price
            support_line_end_date     = resistance_line_end_date
            support_line_end_price    = tmp_support_price
            
            if prices.iloc[idx]['Close'] > tmp_resistance_price or prices.iloc[idx]['Open'] > tmp_resistance_price or \
              prices.iloc[idx]['Close'] < tmp_support_price or prices.iloc[idx]['Open'] < tmp_support_price :
                break
        # 建立型態資料
        pattern_data_dict = {'resistance_line_start_date' : resistance_line_start_date , 'resistance_line_end_date' : resistance_line_end_date , \
                             'resistance_line_start_price' : resistance_line_start_price, 'resistance_line_end_price' : resistance_line_end_price, \
                             'support_line_start_date' : support_line_start_date, 'support_line_end_date' : support_line_end_date, \
                             'support_line_start_price' : support_line_start_price , 'support_line_end_price' : support_line_end_price, \
                             'window':pattern_window}
        pattern_return_dict = {'類型' : '中繼型態', '型態' : pattern_name, '資料' : pattern_data_dict}
        
    else :
        pass
        
    return pattern_return_dict

##### 【內部函式】 型態識別整合 #####
def chart_pattern_recognition( prices, debug=False):
    # 型態回傳列表
    pattern_return_list = []
    
    # 建立「股票技術型態識別」物件
    chart_pattern = StockChartPatterns( prices, debug=debug)
    
    # 識別：底型反轉操作法之底部型態
    bottom_pattern_exist    = False
    turning_point_args      = []
    turning_point_args.append({'mode':'close', 'order':10, 'smoothing':3})
    turning_point_args.append({'mode':'open_close', 'order':5})
    patterns                = chart_pattern.RecognitionBottom(turning_point_args=turning_point_args)
    possible_bottom_pattern = None
    possible_bottom_end_idx = date_to_index(prices,prices.iloc[0].name)
    for pattern in patterns :
        bottom_pattern = patterns[-1]['bottom_pattern']
        bottom_end_idx = date_to_index(prices,bottom_pattern['neckline_end_date'])
        if bottom_end_idx > possible_bottom_end_idx :
            possible_bottom_pattern = bottom_pattern
            possible_bottom_end_idx = bottom_end_idx
    if possible_bottom_pattern is not None :
        if date_to_index(prices,possible_bottom_pattern['neckline_end_date']) > (date_to_index(prices,prices.iloc[-1].name) // 2) :
            bottom_pattern_exist = True
        else :
            pattern_recognition_debug_print('◆ 找到的底部型態太靠前 ： 頸線結束日期 ＝ {} （索引 ＝ {}）， 價格資料的最後一個索引 ＝ {} '.format(possible_bottom_pattern['neckline_end_date'],date_to_index(prices,possible_bottom_pattern['neckline_end_date']),date_to_index(prices,prices.iloc[-1].name)),debug=debug)
    
    # 由轉折點識別型態
    turning_point_args = []
    turning_point_args.append({'mode':'close', 'order':10, 'smoothing':3})
    turning_point_args.append({'mode':'close', 'order':10, 'smoothing':5})
    turning_point_args.append({'mode':'open_close', 'order':10})
    turning_point_args.append({'mode':'open_close', 'order':20})
    patterns         = chart_pattern.Recognition(max_bars=360, turning_point_args=turning_point_args)
    possible_pattern = None
    if patterns is not None and len(patterns) > 0 :
        price_first_idx    = date_to_index(prices,prices.iloc[0].name)
        if bottom_pattern_exist is True:
            price_last_idx = date_to_index(prices,possible_bottom_pattern['neckline_start_date'])
        else :
            price_last_idx = date_to_index(prices,prices.iloc[-1].name)
        for pattern_name in patterns:
            if bottom_pattern_exist is True and ('底' in pattern_name or pattern_name == '下降楔形') :
                continue
            for pattern in patterns[pattern_name] :
                if possible_pattern is None :
                    current_pattern_last_idx = date_to_index(prices,pattern['window'].iloc[-1]['Date'])
                    if current_pattern_last_idx < price_last_idx : 
                        possible_pattern     = [pattern_name,pattern['argument'],pattern['window']]
                else :
                    current_pattern_last_idx   = date_to_index(prices,pattern['window'].iloc[-1]['Date'])
                    possible_pattern_last_idx  = date_to_index(prices,possible_pattern[2].iloc[-1]['Date'])
                    if current_pattern_last_idx < price_last_idx and (price_last_idx - current_pattern_last_idx) < (price_last_idx - possible_pattern_last_idx) :
                        possible_pattern = [pattern_name,pattern['argument'],pattern['window']]
    
    # 處理識別結果
    if possible_pattern is not None :
        pattern_name    = possible_pattern[0]
        pattern_window  = possible_pattern[2]
        result          = pattern_post_processing( prices, pattern_name, pattern_window, debug=debug)
        pattern         = None
        pattern_end_idx = None
        
        if result is not None :
            pattern = result['資料']
            if 'neckline_end_date' in pattern :
                pattern_end_idx = date_to_index(prices,pattern['neckline_end_date'])
            elif 'support_line_end_date' in pattern :
                pattern_end_idx = date_to_index(prices,pattern['support_line_end_date'])
            if pattern_end_idx is not None and pattern_end_idx > (date_to_index(prices,prices.iloc[-1].name) // 2) :
                pattern_return_list.append(result)
            else :
                pattern_recognition_debug_print('◇ 從轉折點找到的型態太靠前 ： 頸線／趨勢線結束日期 ＝ {} （索引 ＝ {}）， 價格資料的最後一個索引 ＝ {} '.format(prices.iloc[pattern_end_idx].name.strftime('%Y-%m-%d'),pattern_end_idx,price_last_idx),debug=debug)
    if bottom_pattern_exist is True and possible_bottom_pattern is not None :
        pattern_return_list.append({'類型' : '底型反轉', '型態' : '底部型態', '資料' : possible_bottom_pattern })
    
    return pattern_return_list

###### 【內部函式】 產生底部型態K線圖的圖像 #####
def generate_bottom_pattern_image( prices, recognition_patterns):
    seq_of_seq_of_points = []
    alines_linewidths    = []
    range_prices         = None
    image                = None
    
    for recognition_pattern in recognition_patterns :
        pattern_type = recognition_pattern['類型']
        pattern_name = recognition_pattern['型態']
        pattern      = recognition_pattern['資料']
        
        if pattern_type == '底型反轉' :
            
            # 讀取底部型態資訊
            bottom_pattern          = pattern
            previous_high_date      = bottom_pattern['previous_high_date']
            bottom_pattern_breakout = False
            if 'bottom_pattern_breakout_date' in bottom_pattern :
                bottom_pattern_breakout_date  = bottom_pattern['bottom_pattern_breakout_date']
                bottom_pattern_breakout_price = bottom_pattern['bottom_pattern_breakout_price']
                bottom_pattern_breakout = True
            neckline_start_date     = bottom_pattern['neckline_start_date']
            neckline_start_price    = bottom_pattern['neckline_start_price']
            neckline_end_date       = bottom_pattern['neckline_end_date']
            neckline_end_price      = bottom_pattern['neckline_end_price']
            bottom_price_date       = bottom_pattern['bottom_price_date']
            bottom_price            = bottom_pattern['bottom_price']
            on_neckline_price       = bottom_pattern['on_neckline_price']
            target_price            = bottom_pattern['target_price']
            
            # 設定頸線
            if bottom_pattern_breakout is True :
                seq_of_seq_of_points.append([(neckline_start_date,neckline_start_price),(neckline_end_date,neckline_end_price)])
                alines_linewidths.append(1.2)
                seq_of_seq_of_points.append([(bottom_price_date,bottom_price),(bottom_price_date,on_neckline_price)])
                alines_linewidths.append(10)
                seq_of_seq_of_points.append([(bottom_pattern_breakout_date,bottom_pattern_breakout_price),(bottom_pattern_breakout_date,target_price)])
                alines_linewidths.append(10)
            else :
                seq_of_seq_of_points.append([(neckline_start_date,neckline_start_price),(neckline_end_date,neckline_end_price)])
                alines_linewidths.append(1.2)
                seq_of_seq_of_points.append([(bottom_price_date,bottom_price),(bottom_price_date,on_neckline_price)])
                alines_linewidths.append(10)
                seq_of_seq_of_points.append([(neckline_end_date,neckline_end_price),(neckline_end_date,target_price)])
                alines_linewidths.append(10)
            
            # 設定圖像區間
            range_prices = prices[previous_high_date:]
            
            break
    
    if range_prices is not None and seq_of_seq_of_points != [] and alines_linewidths != [] :
        # 設定K線格式
        mc = mpf.make_marketcolors(up='xkcd:light red', down='xkcd:almost black', inherit=True)
        s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
        # 設定參數
        img = io.BytesIO()
        kwargs = dict(type='candle', style=s, figratio=(19,10), volume=True, alines=dict(alines=seq_of_seq_of_points, linewidths=tuple(alines_linewidths), colors='xkcd:orange yellow', alpha=0.6), datetime_format='%Y-%m-%d', savefig=dict(fname=img,format='png'))
        # 繪製K線圖
        mpf.plot(range_prices,**kwargs)
        # 刪除空白
        image    = Image.open(img).convert("RGB")
        image    = crop_borders(image, crop_color=(255, 255, 255))
        crop_img = io.BytesIO()
        image.save(crop_img,format='PNG')
    
    return image

###### 【內部函式】 來源 ： https://hackmd.io/@tai-quantup/ch5 #####
# 黃金交叉
def crossover(over,down):
    a1 = over
    b1 = down
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossover =  (a1>a2) & (a1>b1) & (b2>a2)
    return crossover
# 死亡交叉
def crossunder(down,over):
    a1 = down
    b1 = over
    a2 = a1.shift(1)
    b2 = b1.shift(1)
    crossdown =  (a1<a2) & (a1<b1) & (b2<a2)
    return crossdown
    
##### 股票解盤 #####
class StockAnalysis :
    
    ### StockAnalysis類別的建構子 ###
    def __init__( self, sqlite_db_path = 'data/stock.db', debug = False) :
        # 設定除錯旗標
        self._debug = debug
        # 連線資料庫
        if os.path.isfile(sqlite_db_path):
            self._conn = sqlite3.connect( sqlite_db_path)
        else :
            raise ValueError('The database file does not exist.')
        # 從資料庫中載入「台股總覽 TaiwanStockInfo」
        try :
            self._df_stock_info = pd.read_sql("SELECT * FROM StockInfo", self._conn)
        except Exception as e:
            raise ValueError('An error occurred while loading TaiwanStockInfo.')
        # 建立OpenAI API物件
        # 參照 ： https://pypi.org/project/openai/
        load_dotenv(find_dotenv())
        api_key = os.environ.get('OPENAI_API_KEY_TOKEN')
        try :
            self._openai_api_client = OpenAI(api_key = api_key)
        except Exception as e:
            raise ValueError('Failed to create OpenAI object.')
        # 定義內部屬性
        self._reset_internal_attribute()
        
    ### 解盤程序 ###
    def analysis( self, stock_id) :
        
        # 技術分析工具列表
        tool_list = []
        
        # 解盤內容
        description_list = []
        
        # 重置內部屬性
        self._reset_internal_attribute()
        
        # 從資料庫中載入價格資料
        result = self._loading_price_data(stock_id)
        if result is False :
            return None
        
        # 技術指標計算
        self._technical_indicators()
        
        # （１） Ｋ線／Ｋ棒
        description = self._k_line_pattern_recogntion()
        tool_list.append('Ｋ線／Ｋ棒')
        description_list.append(description)
        
        # （２） Ｋ線圖
        description = self._rank_evaluate()
        tool_list.append('Ｋ線圖')
        description_list.append(description)
        
        # （３） 成交量
        description = self._price_volume_relationship()
        tool_list.append('成交量')
        description_list.append(description)
        
        # （４） 型態
        description, bottom_pattern_image = self._pattern_recognition()
        tool_list.append('型態')
        description_list.append(description)
        if bottom_pattern_image is not None :
            self._image_dict['底部型態'] = bottom_pattern_image
        
        # （５） 移動平均線
        description = self._moving_average_trend()
        tool_list.append('移動平均線')
        description_list.append(description)
        
        # （６） ＫＤ指標
        description = self._kd_cross()
        tool_list.append('ＫＤ指標')
        description_list.append(description)
        
        # （７） ＭＡＣＤ指標
        description = self._macd_long_short()
        tool_list.append('ＭＡＣＤ指標')
        description_list.append(description)
        
        # （８） 周Ｋ－周ＫＤ指標
        description = self._weekly_kd_cross()
        tool_list.append('周Ｋ－周ＫＤ指標')
        description_list.append(description)
        
        # 整體評價
        overall_evaluation_image,overall_evaluation_result = self._overall_evaluation()
        tool_list.append('整體評價')
        description_list.append(overall_evaluation_result)
        self._image_dict['整體評價'] = overall_evaluation_image
        
        # 解盤內容輸出
        df_result         = pd.DataFrame([tool_list, description_list])
        df_result         = df_result.transpose()
        df_result.columns = ['技術分析工具', '解盤內容']
        df_result.set_index(df_result['技術分析工具'],inplace=True)
        df_result         = df_result.drop(columns=['技術分析工具'])
        
        # 輸出：解盤內容(DataFrame),底部型態影像(dict)
        return df_result,self._image_dict
        
    ### 列印除錯訊息之內部方法 ###
    def _debug_print( self, msg) :
        if self._debug is True :
            print("ＤＥＢＵＧ ： {}".format(msg))
    
    ### 重置內部屬性之內部方法 ###
    def _reset_internal_attribute( self) :
        self._daily_price_df        = None
        self._weekly_price_df       = None
        self._daily_price_talib_df  = None
        self._sma_df                = None
        self._kd_df                 = None
        self._macd_df               = None
        self._weekly_kd_df          = None
        self._price_volume_unit_str = None
        self._price_unit            = None
        self._volume_unit           = None
        self._image_dict            = {}
    
    ### 從資料庫中載入日Ｋ與周Ｋ資料之內部方法 ###
    def _loading_price_data( self, stock_id) :
        # 將載入的「台股總覽 TaiwanStockInfo」進行格式轉換
        df_stock_info = self._df_stock_info.set_index(self._df_stock_info['StockID'],inplace=False)
        df_stock_info = df_stock_info.drop(columns=['StockID'])
        
        # 判斷股票代碼(stock_id)是否存在於「台股總覽 TaiwanStockInfo」中
        if stock_id in df_stock_info.index:
            
            # 取得該股票代碼的產業分類
            individual_stock_info = df_stock_info.loc[stock_id]
            industry_category     = individual_stock_info['IndustryCategory']
            
            # 設定開始日期與結束日期
            current_date       = datetime.datetime.today()
            daily_end_date     = current_date.strftime('%Y-%m-%d')
            daily_start_date,_ = get_monday_to_sunday((current_date - datetime.timedelta(days=730)).strftime('%Y-%m-%d'))
            weekly_start_date  = daily_start_date
            weekly_end_date,_  = get_monday_to_sunday(daily_end_date)
            sqlcmd             = "SELECT * FROM WeeklyPrice WHERE Date='{}'".format(weekly_end_date)
            df                 = pd.read_sql_query(sqlcmd, self._conn)
            if df.empty is True :
                weekly_end_date,_ = get_monday_to_sunday(daily_end_date,weekly=-1)
            self._debug_print('日Ｋ開始日期 ＝ {} ，日Ｋ結束日期 ＝ {} ， 周Ｋ開始日期 ＝ {} ， 周Ｋ結束日期 ＝ {}'.format(daily_start_date,daily_end_date,weekly_start_date,weekly_end_date))
            
            # 讀取日Ｋ價格資料
            sql_cmd = "SELECT * FROM DailyPrice WHERE StockID='{}' AND (Date BETWEEN '{}' AND '{}')".format(stock_id,daily_start_date,daily_end_date)
            try :
                daily_price_df = pd.read_sql( sql_cmd, self._conn)
            except Exception as e:
                self._debug_print('讀取日K資料錯誤，錯誤訊息＝ {}'.format(str(e)))
                return False
            # 格式轉換：日期格式、成交量(成交值)
            daily_price_df           = daily_price_df.drop(columns=['SerialNo','StockID'])
            daily_price_df['Date']   = daily_price_df['Date'].astype('datetime64[ns]')
            daily_price_df.set_index(daily_price_df['Date'],inplace=True)
            daily_price_df           = daily_price_df.drop(columns=['Date'])
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
            self._daily_price_df         = daily_price_df
            
            # 讀取周Ｋ價格資料
            sql_cmd = "SELECT * FROM WeeklyPrice WHERE StockID='{}' AND (Date BETWEEN '{}' AND '{}')".format(stock_id,weekly_start_date,weekly_end_date)
            try :
                weekly_price_df = pd.read_sql( sql_cmd, self._conn)
            except Exception as e:
                self._debug_print('讀取周K資料錯誤，錯誤訊息＝ {}'.format(str(e)))
                return False
            # 格式轉換：日期格式、成交量(成交值)
            weekly_price_df           = weekly_price_df.drop(columns=['SerialNo','StockID'])
            weekly_price_df['Date']   = weekly_price_df['Date'].astype('datetime64[ns]')
            weekly_price_df.set_index(weekly_price_df['Date'],inplace=True)
            weekly_price_df           = weekly_price_df.drop(columns=['Date'])
            if industry_category == 'Index' or industry_category == '大盤' :
                weekly_price_df           = weekly_price_df.drop(columns=['Volume'])
                weekly_price_df           = weekly_price_df.rename(columns={'Value':'Volume'})
                weekly_price_df['Volume'] = weekly_price_df['Volume'].div(100000000.00)
                weekly_price_df['Volume'] = weekly_price_df['Volume'].round(2)
            else :
                weekly_price_df           = weekly_price_df.drop(columns=['Value'])
                weekly_price_df['Volume'] = weekly_price_df['Volume'].div(1000)
                weekly_price_df['Volume'] = weekly_price_df['Volume'].round()
                weekly_price_df['Volume'] = weekly_price_df['Volume'].astype('int64')
            self._weekly_price_df         = weekly_price_df
            
            if industry_category == 'Index' or industry_category == '大盤' :
                self._price_volume_unit_str = '價格單位為點，成交量單位為億元'
                self._price_unit            = '點'
                self._volume_unit           = '億元'
            else :
                self._price_volume_unit_str = '價格單位為元，成交量單位為張'
                self._price_unit            = '元'
                self._volume_unit           = '張'
                
            return True
        return False
    
    ### 使用talib程式庫計算技術指標之內部方法 ###
    def _technical_indicators( self) :
        # 日Ｋ價格資料轉換為talib格式
        daily_price_df_talib          = self._daily_price_df.copy()
        daily_price_df_talib.columns  = [ i.lower() for i in daily_price_df_talib.columns]
        self._daily_price_talib_df    = daily_price_df_talib
        
        # 周Ｋ價格資料轉換為talib格式
        weekly_price_df_talib         = self._weekly_price_df.copy()
        weekly_price_df_talib.columns = [ i.lower() for i in weekly_price_df_talib.columns]
        
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
        self._sma_df      = talib_sma_df.round(2)
        
        # 計算ＫＤ指標
        talib_daily_kd = STOCH( daily_price_df_talib, fastk_period=9, slowk_period=3, slowd_period=3)
        # 取小數點後兩位
        self._kd_df    = talib_daily_kd.round(2)
        
        # 計算ＭＡＣＤ指標
        talib_daily_macd = MACD( daily_price_df_talib, fastperiod=12, slowperiod=26, signalperiod=9)
        # 取小數點後兩位
        self._macd_df    = talib_daily_macd.round(2)
        
        # 計算周ＫＤ指標
        talib_weekly_kd    = STOCH( weekly_price_df_talib, fastk_period=6, slowk_period=3, slowd_period=3)
        # 取小數點後兩位
        self._weekly_kd_df = talib_weekly_kd.round(2)
        
    ### 量化技術分析工具： Ｋ線型態識別 ###
    def _k_line_pattern_recogntion( self) :
        # 設定K線型態資訊
        k_line_patterns = defaultdict(list)
            
        # 識別紡錘線型態
        patterns       = CDLSPINNINGTOP(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['紡錘線'].append(index.strftime("%Y-%m-%d"))
            
        # 識別十字線(含變形)型態
        patterns       = CDLDOJI(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['十字線'].append(index.strftime("%Y-%m-%d"))
        patterns       = CDLDRAGONFLYDOJI(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value !=0 :
                if index.strftime("%Y-%m-%d") not in k_line_patterns['十字線'] :
                    k_line_patterns['十字線'].append(index.strftime("%Y-%m-%d"))
            
        # 識別鎚子線／吊人線型態
        patterns       = CDLHAMMER(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['鎚子線／吊人線'].append(index.strftime("%Y-%m-%d"))
        patterns       = CDLHANGINGMAN(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                if index.strftime("%Y-%m-%d") not in k_line_patterns['鎚子線／吊人線'] :
                    k_line_patterns['鎚子線／吊人線'].append(index.strftime("%Y-%m-%d"))
            
        # 識別墓碑線型態
        patterns       = CDLINVERTEDHAMMER(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['墓碑線'].append(index.strftime("%Y-%m-%d"))
        
        # 識別吞噬型態
        patterns       = CDLENGULFING(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['吞噬'].append(index.strftime("%Y-%m-%d"))
            
        # 識別孕育線／懷抱線型態
        patterns       = CDLHARAMI(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['孕育線／懷抱線'].append(index.strftime("%Y-%m-%d"))
        patterns       = CDLHARAMICROSS(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                if index.strftime("%Y-%m-%d") not in k_line_patterns['孕育線／懷抱線'] :
                    k_line_patterns['孕育線／懷抱線'].append(index.strftime("%Y-%m-%d"))
        
        # 識別高檔夜星型態
        patterns       = CDLEVENINGDOJISTAR(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value!= 0 :
                k_line_patterns['高檔夜星'].append(index.strftime("%Y-%m-%d"))
        patterns      = CDLEVENINGSTAR(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                if index.strftime("%Y-%m-%d") not in k_line_patterns['高檔夜星'] :
                    k_line_patterns['高檔夜星'].append(index.strftime("%Y-%m-%d"))
        
        # 識別上升三法型態
        patterns       = CDLRISEFALL3METHODS(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['上升三法'].append(index.strftime("%Y-%m-%d"))
        
        # 識別紅三兵型態
        patterns       = CDL3WHITESOLDIERS(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0 :
                k_line_patterns['紅三兵'].append(index.strftime("%Y-%m-%d"))
        
        # 識別黑三兵型態
        patterns       = CDL3BLACKCROWS(self._daily_price_talib_df)
        range_patterns = patterns[-5:]
        for index, value in range_patterns.items():
            if value != 0:
                k_line_patterns['黑三兵'].append(index.strftime("%Y-%m-%d"))
                
        description_str = ''
        for pattern_name in k_line_patterns :
            description_str = description_str + pattern_name + ' : '
            for pattern_date in k_line_patterns[pattern_name] :
                description_str = description_str + '{} , '.format(pattern_date)
            description_str = description_str.rstrip(' ,')
            description_str = description_str + '\n'
        
        return description_str
    
    ### 量化技術分析工具： 量化位階評價 ###
    def _rank_evaluate( self) :
        
        # 解盤內容
        description_str = ''
        
        # 設定中期(60)區間價格
        range_close_prices = np.array(self._daily_price_df[-60:]['Close'])
        # 計算平均值與標準差
        base_prices_mean   = np.mean(range_close_prices)
        base_prices_std    = np.std(range_close_prices)
        # 位階評價
        price = range_close_prices[-1]
        rank  = ''
        price_std_range = (price - base_prices_mean) / base_prices_std
        if price_std_range <= -1.0 :
            rank = '低' 
        elif price_std_range > -1.0 and price_std_range < 1.0 :
            rank = '中'
        elif price_std_range >= 1.0:
            rank = '高'
            
        description_str = description_str + '中期為{}位階\n'.format(rank)
            
        # 設定長期(240)區間價格
        range_close_prices = np.array(self._daily_price_df[-240:]['Close'])
        # 計算平均值與標準差
        base_prices_mean   = np.mean(range_close_prices)
        base_prices_std    = np.std(range_close_prices)
        # 位階評價
        rank  = ''
        price = range_close_prices[-1]
        price_std_range = (price - base_prices_mean) / base_prices_std
        if price_std_range <= -1.0 :
            rank = '低' 
        elif price_std_range > -1.0 and price_std_range < 1.0 :
            rank = '中'
        elif price_std_range >= 1.0:
            rank = '高'
            
        description_str = description_str + '長期為{}位階'.format(rank)
        
        return description_str
        
    ### 量化技術分析工具： 價量關係 ###
    def _price_volume_relationship( self) :
        # 解盤內容
        description_str = '價量關係：'
        
        # 計算量的移動平均 － 十日均量
        volume_ma10 = self._daily_price_df['Volume'].rolling(window=10).mean()
        if float.is_integer(self._daily_price_df.iloc[0]['Volume']) is True :
            volume_ma10 = volume_ma10.round()
            volume_ma10 = volume_ma10.to_numpy()
            volume_ma10 = np.nan_to_num(volume_ma10)
            volume_ma10 = volume_ma10.astype('int64')
        else:
            volume_ma10 = volume_ma10.round(2)
            volume_ma10 = volume_ma10.to_numpy()
        
        # 價量關係判斷
        volume         = self._daily_price_df['Volume'].to_numpy()
        current_volume = volume[-1]
        current_ma10   = volume_ma10[-1]
        
        # 大量：比10日均量多出30%
        check_volume_1 = current_ma10 + ((current_ma10 * 30) / 100)
        # 爆量：比10日均量多出50%
        check_volume_2 = current_ma10 + ((current_ma10 * 50) / 100)
        # 天量：比10日均量多出100%
        check_volume_3 = current_ma10 * 2

        # 大量：10日均量的3倍
        check_volume_4 = current_ma10 * 3
        # 爆量：10日均量的5倍
        check_volume_5 = current_ma10 * 5
        # 天量：10日均量多10倍
        check_volume_6 = current_ma10 * 10

        if current_volume >= check_volume_6 :
            description_str = description_str + '中小型股天量／大型股天量'
        elif current_volume >= check_volume_5 :
            description_str = description_str + '中小型股爆量／大型股天量'
        elif current_volume >= check_volume_4 :
            description_str = description_str + '中小型股大量／大型股天量'
        elif current_volume >= check_volume_3 :
            description_str = description_str + '大型股天量'
        elif current_volume >= check_volume_2 :
            description_str = description_str + '大型股爆量'
        elif current_volume >= check_volume_1 :
            description_str = description_str + '大型股大量'
        else :
            description_str = description_str + '未至大量門檻'
        description_str = description_str + '（成交量 ＝ {}{} ，十日均量 ＝ {}{}）'.format(current_volume,self._volume_unit,current_ma10,self._volume_unit)
        
        return description_str
    
    ### 量化技術分析工具： 價格型態識別 ###
    def _pattern_recognition( self) :
        # 設定長期(240+120)區間價格
        range_prices = self._daily_price_df[-360:]
        
        # 圖形型態識別
        result = chart_pattern_recognition( range_prices, debug = self._debug)
        
        # 識別結果轉換成文字敘述
        description_str = ''
        for pattern in result :
            if 'is_breakout' in pattern['資料'] or 'bottom_pattern_breakout_date' in pattern['資料'] :
                pattern_description = '{}之{}，型態範圍由{}開始到{}結束，估算目標價為{:.2f}{}。\n'.format(pattern['類型'],pattern['型態'],pattern['資料']['neckline_start_date'],pattern['資料']['neckline_end_date'],pattern['資料']['target_price'],self._price_unit)
            else :
                if 'neckline_start_date' in pattern['資料'] and 'neckline_end_date' in pattern['資料'] :
                    pattern_description = '尚未成形{}之{}，型態範圍由{}開始到{}結束。\n'.format(pattern['類型'],pattern['型態'],pattern['資料']['neckline_start_date'],pattern['資料']['neckline_end_date'])
                else :
                    pattern_description = '{}之{}，型態範圍由{}開始到{}結束。\n'.format(pattern['類型'],pattern['型態'],pattern['資料']['support_line_start_date'],pattern['資料']['support_line_end_date'])
            description_str = description_str + pattern_description
        
        # 產生底部型態圖像
        bottom_pattern_image = generate_bottom_pattern_image(range_prices, result)
        
        return description_str,bottom_pattern_image
        
    ### 量化技術分析工具： 使用移動平均線判斷趨勢 ###
    def _moving_average_trend( self) :
        # 解盤內容
        description_str = ''
        
        now_talib_daily_sma  = self._sma_df.iloc[-1].to_dict()
        prev_talib_daily_sma = self._sma_df.iloc[-2].to_dict()
        # 確認移動平均線方向：↑ : 'u' , ↓ : 'd' , ＝ : '='
        talib_daily_sma5_dir = 'u' if now_talib_daily_sma['SMA5'] > prev_talib_daily_sma['SMA5'] else 'd' if now_talib_daily_sma['SMA5'] < prev_talib_daily_sma['SMA5'] else '='
        talib_daily_sma10_dir = 'u' if now_talib_daily_sma['SMA10'] > prev_talib_daily_sma['SMA10'] else 'd' if now_talib_daily_sma['SMA10'] < prev_talib_daily_sma['SMA10'] else '='
        talib_daily_sma20_dir = 'u' if now_talib_daily_sma['SMA20'] > prev_talib_daily_sma['SMA20'] else 'd' if now_talib_daily_sma['SMA20'] < prev_talib_daily_sma['SMA20'] else '='
        talib_daily_sma60_dir = 'u' if now_talib_daily_sma['SMA60'] > prev_talib_daily_sma['SMA60'] else 'd' if now_talib_daily_sma['SMA60'] < prev_talib_daily_sma['SMA60'] else '='
        talib_daily_sma120_dir = 'u' if now_talib_daily_sma['SMA120'] > prev_talib_daily_sma['SMA120'] else 'd' if now_talib_daily_sma['SMA120'] < prev_talib_daily_sma['SMA120'] else '='
        talib_daily_sma240_dir = 'u' if now_talib_daily_sma['SMA240'] > prev_talib_daily_sma['SMA240'] else 'd' if now_talib_daily_sma['SMA240'] < prev_talib_daily_sma['SMA240'] else '='
        # 短期趨勢
        if talib_daily_sma5_dir == 'u' :
            description_str = description_str + '5日線上揚'
        elif talib_daily_sma5_dir == 'd' :
            description_str = description_str + '5日線下彎'
        else :
            description_str = description_str + '5日線走平'
        description_str = description_str + '、'
        if talib_daily_sma10_dir == 'u' :
            description_str = description_str + '10日線上揚'
        elif talib_daily_sma10_dir == 'd' :
            description_str = description_str + '10日線下彎'
        else :
            description_str = description_str + '10日線走平'
        description_str = description_str + '，'
        if talib_daily_sma5_dir == talib_daily_sma10_dir :
            if talib_daily_sma5_dir == 'u' :
                description_str = description_str + '短期翻多'
            elif talib_daily_sma5_dir == 'd' :
                description_str = description_str + '短期翻空'
            else :
                description_str = description_str + '短期盤整'
        else :
            description_str = description_str + '短期盤整'
        description_str = description_str + '；'
        # 中期趨勢
        if talib_daily_sma20_dir == 'u' :
            description_str = description_str + '20日線上揚'
        elif talib_daily_sma20_dir == 'd' :
            description_str = description_str + '20日線下彎'
        else :
            description_str = description_str + '20日線走平'
        description_str = description_str + '、'
        if talib_daily_sma60_dir == 'u' :
            description_str = description_str + '60日線上揚'
        elif talib_daily_sma60_dir == 'd' :
            description_str = description_str + '60日線下彎'
        else :
            description_str = description_str + '60日線走平'
        description_str = description_str + '，'
        if talib_daily_sma20_dir == talib_daily_sma60_dir :
            if talib_daily_sma20_dir == 'u' :
                description_str = description_str + '中期翻多'
            elif talib_daily_sma20_dir == 'd' :
                description_str = description_str + '中期翻空'
            else :
                description_str = description_str + '中期盤整'
        else :
            description_str = description_str + '中期盤整'
        description_str = description_str + '；'
        # 長期趨勢
        if talib_daily_sma120_dir == 'u' :
            description_str = description_str + '120日線上揚'
        elif talib_daily_sma120_dir == 'd' :
            description_str = description_str + '120日線下彎'
        else :
            description_str = description_str + '120日線走平'
        description_str = description_str + '、'
        if talib_daily_sma240_dir == 'u' :
            description_str = description_str + '240日線上揚'
        elif talib_daily_sma240_dir == 'd' :
            description_str = description_str + '240日線下彎'
        else :
            description_str = description_str + '240日線走平'
        description_str = description_str + '，'
        if talib_daily_sma120_dir == talib_daily_sma240_dir :
            if talib_daily_sma120_dir == 'u' :
                description_str = description_str + '長期翻多'
            elif talib_daily_sma120_dir == 'd' :
                description_str = description_str + '長期翻空'
            else :
                description_str = description_str + '長期盤整'
        else :
            description_str = description_str + '長期盤整'
            
        return description_str
        
    ### 量化技術分析工具： 確認ＫＤ指標交叉 ###
    def _kd_cross( self) :

        # 設定中期(60)區間價格與均線
        range_prices = self._daily_price_df[-60:]

        # 設定中期(60)日KD指標區間
        range_talib_daily_kd = self._kd_df[-60:]
        
        # 尋找KD黃金交叉
        ret_over  = crossover(range_talib_daily_kd['slowk'],range_talib_daily_kd['slowd'])
        
        # 尋找KD死亡交叉
        ret_under = crossunder(range_talib_daily_kd['slowk'],range_talib_daily_kd['slowd'])
        
        # 解盤內容
        description_str = ''
        
        # 最後一個交叉點確認
        golden_cross_last_date = None
        death_cross_last_date  = None
        for idx in range(0,len(ret_over)) :
            if ret_over.iloc[idx]:
                golden_cross_last_date = range_talib_daily_kd.iloc[idx].name.strftime('%Y-%m-%d')

        for idx in range(0,len(ret_under)) :
            if ret_under.iloc[idx]:
                death_cross_last_date  = range_talib_daily_kd.iloc[idx].name.strftime('%Y-%m-%d')
        
        if golden_cross_last_date is not None and death_cross_last_date is not None:
            golden_cross_last_index = date_to_index(range_prices,golden_cross_last_date)
            death_cross_last_index  = date_to_index(range_prices,death_cross_last_date)
            if golden_cross_last_index > death_cross_last_index :
                description_str = description_str + '於{}黃金交叉。'.format(golden_cross_last_date)
            else :
                description_str = description_str + '於{}死亡交叉。'.format(death_cross_last_date)
        else :
            if golden_cross_last_date is not None :
                description_str = description_str + '於{}黃金交叉。'.format(golden_cross_last_date)
            elif death_cross_last_date is not None :
                description_str = description_str + '於{}死亡交叉。'.format(death_cross_last_date)
            else :
                description_str = description_str + '未發生交叉。'.format(death_cross_last_date)
            
        # 超買區與超賣區判斷
        if range_talib_daily_kd['slowk'].iloc[-1] > 80.0 and range_talib_daily_kd['slowd'].iloc[-1] > 80.0 :
            description_str = description_str + '並且{}於超買區。'.format(range_talib_daily_kd.iloc[-1].name.strftime('%Y-%m-%d'))
        elif range_talib_daily_kd['slowk'].iloc[-1] < 20.0 and range_talib_daily_kd['slowd'].iloc[-1] < 20.0:
            description_str = description_str + '並且{}於超賣區。'.format(range_talib_daily_kd.iloc[-1].name.strftime('%Y-%m-%d'))
         
        return description_str
    
    ### 量化技術分析工具： ＭＡＣＤ指標確認中期多空 ###
    def _macd_long_short( self) :
        # 解盤內容
        description_str = ''
        
        if self._macd_df['macdsignal'].iloc[-1] > 0 :
            description_str = description_str + 'MACD在零軸以上，中期多方'
        elif self._macd_df['macdsignal'].iloc[-1] < 0 :
            description_str = description_str + 'MACD在零軸以下，中期空方'
        
        return description_str
        
    ### 量化技術分析工具： 確認周ＫＤ指標交叉 ###
    def _weekly_kd_cross( self) :
        # 設定中期(60)周K線區間
        start_date             = self._daily_price_talib_df.iloc[-60].name.strftime('%Y-%m-%d')
        end_date               = self._daily_price_talib_df.iloc[-1].name.strftime('%Y-%m-%d')
        weekly_start_date,_    = get_monday_to_sunday(start_date)
        weekly_end_date,_      = get_monday_to_sunday(end_date)
        sqlcmd                 = "SELECT * FROM WeeklyPrice WHERE Date='{}'".format(weekly_end_date)
        df                     = pd.read_sql_query(sqlcmd, self._conn)
        if df.empty is True :
            weekly_end_date,_  = get_monday_to_sunday(end_date,weekly=-1)
        self._debug_print('周K線範圍 ： 開始日期 ＝ {} ， 結束日期 ＝ {}'.format(weekly_start_date,weekly_end_date))

        # 設定中期(60)周K線範圍
        range_weekly_price_df  = self._weekly_price_df[weekly_start_date:weekly_end_date]

        # 設定中期(60)周KD指標區間
        range_talib_weekly_kd  = self._weekly_kd_df[weekly_start_date:weekly_end_date]
        
        # 尋找KD黃金交叉
        ret_over = crossover(range_talib_weekly_kd['slowk'],range_talib_weekly_kd['slowd'])
        
        # 尋找KD死亡交叉
        ret_under = crossunder(range_talib_weekly_kd['slowk'],range_talib_weekly_kd['slowd'])
        
        # 解盤內容
        description_str = ''
        
        # 最後一個交叉點確認
        golden_cross_last_date = None
        death_cross_last_date  = None
        
        for idx in range(0,len(ret_over)) :
            if ret_over.iloc[idx]:
                golden_cross_last_date = range_talib_weekly_kd.iloc[idx].name.strftime('%Y-%m-%d')
        for idx in range(0,len(ret_under)) :
            if ret_under.iloc[idx]:
                death_cross_last_date  = range_talib_weekly_kd.iloc[idx].name.strftime('%Y-%m-%d')
        
        if golden_cross_last_date is not None and death_cross_last_date is not None:
            golden_cross_last_index = date_to_index(range_weekly_price_df,golden_cross_last_date)
            death_cross_last_index  = date_to_index(range_weekly_price_df,death_cross_last_date)

            if golden_cross_last_index > death_cross_last_index :
                description_str = description_str + '於{}當周黃金交叉。'.format(golden_cross_last_date)
            else :
                description_str = description_str + '於{}當周死亡交叉。'.format(death_cross_last_date)
        else :
            if golden_cross_last_date is not None :
                description_str = description_str + '於{}當周黃金交叉。'.format(golden_cross_last_date)
            elif death_cross_last_date is not None :
                description_str = description_str + '於{}當周死亡交叉。'.format(death_cross_last_date)
            else :
                description_str = description_str + '未發生交叉。'.format(death_cross_last_date)
        
        # 超買區與超賣區判斷
        if range_talib_weekly_kd['slowk'].iloc[-1] > 80.0 and range_talib_weekly_kd['slowd'].iloc[-1] > 80.0 :
            description_str = description_str + '並且{}該周在超買區。'.format(range_talib_weekly_kd.iloc[-1].name.strftime('%Y-%m-%d'))
        elif range_talib_weekly_kd['slowk'].iloc[-1] < 20.0 and range_talib_weekly_kd['slowd'].iloc[-1] < 20.0:
            description_str = description_str + '並且{}該周在超賣區。'.format(range_talib_weekly_kd.iloc[-1].name.strftime('%Y-%m-%d'))
        
        return description_str
        
    ### 整體評價 ###
    # 參考：How to build Technical Analyst AI Agent using LLM Vision (no-code with n8n)! https://www.youtube.com/watch?v=yjBHheCB6Ek
    # 參考：市場已進入 AI 判讀時代！30秒技術圖表自動分析：再不升級，投資只能碰運氣！Flowise + n8n 實戰整合 https://www.youtube.com/watch?v=JvhNF86PYJY
    def _overall_evaluation( self) :
        
        range_prices = self._daily_price_df[-60:]
        range_sma    = self._sma_df[-60:]
        range_kd     = self._kd_df[-60:]
        range_macd   = self._macd_df[-60:]
        
        # 設定K線格式
        mc = mpf.make_marketcolors(up='xkcd:light red', down='xkcd:almost black', inherit=True)
        s  = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
        
        # ＫＤ指標：超買線
        kd_overbought_line = [80] * range_prices.shape[0]
        # ＫＤ指標：賣超線
        kd_oversold_line   = [20] * range_prices.shape[0]

        # 設定技術指標
        added_plots = [
            mpf.make_addplot(range_sma['SMA5'],width=0.8,color='xkcd:red brown'),
            mpf.make_addplot(range_sma['SMA10'],width=0.8,color='xkcd:dark sky blue'),
            mpf.make_addplot(range_sma['SMA20'],width=0.8,color='xkcd:violet'),
            mpf.make_addplot(range_sma['SMA60'],width=0.8,color='xkcd:orange'),
            mpf.make_addplot(range_kd['slowk'],width=0.8,panel=2,secondary_y=False,color='xkcd:red',ylabel='KD'),
            mpf.make_addplot(range_kd['slowd'],width=0.8,panel=2,secondary_y=False,color='xkcd:blue'),
            mpf.make_addplot(kd_overbought_line,width=0.8,panel=2,secondary_y=False,linestyle='--',color='xkcd:green'),
            mpf.make_addplot(kd_oversold_line,width=0.8,panel=2,secondary_y=False,linestyle='--',color='xkcd:green'),
            mpf.make_addplot(range_macd['macdhist'],type='bar',panel=3,secondary_y=False,color='xkcd:grey',ylabel='MACD'),
            mpf.make_addplot(range_macd['macd'],width=0.8,panel=3,secondary_y=False,color='xkcd:red'),
            mpf.make_addplot(range_macd['macdsignal'],width=0.8,panel=3,secondary_y=False,color='xkcd:blue')
        ]

        # 設定K線圖X軸座標值
        ticks        = []
        tlabs        = []
        label_count  = 0
        total_k_line = range_prices.shape[0]
        for idx in range(total_k_line) :
            timestamp = range_prices.iloc[idx].name
            if idx == 0 :
                ticks.append(idx)
                tlabs.append(timestamp.strftime('%Y-%m-%d'))
                label_count = 0
            elif (idx+1) == total_k_line:
                if label_count < 5 :
                    # 移除上一筆資料
                    ticks.pop()
                    tlabs.pop()
                ticks.append(idx)
                tlabs.append(timestamp.strftime('%Y-%m-%d'))
            elif label_count > 10 :
                ticks.append(idx)
                tlabs.append(timestamp.strftime('%Y-%m-%d'))
                label_count = 0
            label_count += 1
        
        # 繪製K線圖
        tmp = io.BytesIO()
        kwargs = dict(type='candle', style=s, figratio=(16,9), volume=True, addplot=added_plots, main_panel=0, volume_panel=1, panel_ratios=(5, 1, 2, 2), num_panels=4, datetime_format='%Y-%m-%d', returnfig=True, savefig=dict(fname=tmp))
        fig, axlist = mpf.plot(range_prices,**kwargs)
        axlist[-2].set_xticks(ticks,labels=tlabs,ha='right')
        tmp = None
        
        # 保存K線圖
        img = io.BytesIO()
        fig.savefig(img,format='png')

        # 刪除空白
        image    = Image.open(img).convert("RGB")
        image    = crop_borders(image, crop_color=(255, 255, 255))
        crop_img = io.BytesIO()
        image.save(crop_img,format='PNG')
        
        # 將影像檔(PNG)編碼為Base64
        base64_image = base64.b64encode(crop_img.getvalue()).decode()
        
        # 支撐與壓力判斷之參考資料
        ref_price_sma_df = pd.concat([range_prices[-5:],range_sma[-5:]],axis=1)
        ref_price_sma_df.index = ref_price_sma_df.index.strftime('%Y-%m-%d')
        ref_price_sma_json = ref_price_sma_df.to_json(date_format='iso')
        
        # ＫＤ指標判讀之參考資料
        ref_kd_df   = range_kd[-5:].copy()
        ref_kd_df.index = ref_kd_df.index.strftime('%Y-%m-%d')
        ref_kd_json = ref_kd_df.to_json(date_format='iso')
        
        # ＭＡＣＤ指標判讀之參考資料
        ref_macd_df   = range_macd[-5:].copy()
        ref_macd_df.index = ref_macd_df.index.strftime('%Y-%m-%d')
        ref_macd_json = ref_macd_df.to_json(date_format='iso')
        
        # 系統提示詞（System Prompt） 與 使用者提問詞（User Prompt）
        # TODO：持續驗證與確認中
        system_prompt = '你是一位具備專業技術分析能力的股市分析師，請依據以下圖表進行全面分析並提供交易建議。圖表包含以下元素：主圖：K線圖表示股價趨勢，紅K代表收盤價高於開盤價、黑K代表收盤價低於開盤價，圖中包含四條移動平均線：棕色線為 5日移動平均線（代表短期趨勢）、天藍色線為 10日移動平均線（代表短期趨勢）、紫色線為 20日移動平均線（代表中期趨勢）、橙色線為 60日移動平均線（代表中期趨勢）；子圖一（Volume）：以柱狀體顯示每日成交量，顏色與K線同步反映多空；子圖二（KD 指標）：紅線為 K 線（slowk）、藍線為 D 線（slowd）、綠色虛線區隔超買區（>80）與超賣區（<20）；子圖三（MACD 指標）：紅線為 DIF 線（macd）、藍線為 MACD 線（macdsignal）、灰色柱體為 OSC（macdhist）。你需要結合 價格走勢、成交量變化、KD 與 MACD 指標，判斷：當前的趨勢狀況（上漲/盤整/下跌）、價量是否配合（多頭/空頭）、支撐與壓力位置、短中期可能的走勢、並且綜合分析結果來給予評價。使用通順的繁體中文回覆，字詞代換：回調 -> 回檔，止盈 -> 停利，止損 -> 停損 ...。'
        user_question = '請依據下圖（圖中價格單位為： {} ）進行技術分析，包含價格趨勢（上漲/盤整/下跌）、價量關係、判斷支撐與壓力價格或點位（以上項目若有需要時可參考最近五個交易日的價格與移動平均線資料 ： {} ，但回覆時不要提及所參考資料）、KD指標解讀（解讀時若有需要可參考最近五個交易日的KD指標資料 ： {} ，但回覆時不要提及所參考資料）與MACD指標解讀（解讀時若有需要可參考最近五個交易日的MACD指標資料 ： {} ，但回覆時不要提及所參考資料），並且綜合上述分析結果給出評價。'.format(self._price_volume_unit_str,ref_price_sma_json,ref_kd_json,ref_macd_json)
        
        # 與GPT-4o模型對話
        response = self._openai_api_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_question},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
        )
        
        text      = response.choices[0].message.content
        pure_text = re.sub(r'[*_`#\[\]()~]', '', text)
        
        return image,pure_text
        
    ### StockAnalysis類別的解構子 ###
    def __del__( self) :
        # 關閉資料庫
        self._conn.close()