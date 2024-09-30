#
# @file PerceptuallyImportant.py
#

import pandas as pd
import numpy as np

##### 來源 ： https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/perceptually_important.py #####

def find_pips(data: np.array, n_pips: int, dist_measure: int):
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance

    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                
                d = 0.0 # Distance
                if dist_measure == 1: # Euclidean distance
                    d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                    d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                elif dist_measure == 2: # Perpindicular distance
                    d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                else: # Vertical distance    
                    d = abs( (slope * i + intercept) - data[i] )

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y
    
    
##### 尋找感知重要點函式 #####
# prices       : 價格 (DataFrame)
# mode         : 模式 (str)
# n_pips       : 感知重要點數量 (int)
# dist_measure : 距離測量模式 (str)
def FindingPerceptuallyImportantPoints( prices, mode='close', n_pips=5, dist_measure='Perpindicular') :
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
    if mode == 'close' :
        proc_prices=np.array(in_prices['Close'])
    elif mode == 'open' :
        proc_prices=np.array(in_prices['Open'])
    elif mode == 'high' :
        proc_prices=np.array(in_prices['High'])
    elif mode == 'low' :
        proc_prices=np.array(in_prices['Low'])
    else :
        return None
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance
    if dist_measure == 'Euclidean':
        proc_dist_measure = 1
    elif dist_measure == 'Perpindicular':
        proc_dist_measure = 2
    elif dist_measure == 'Vertical':
        proc_dist_measure = 3
    else :
        return None
    pips_x, pips_y = find_pips(data=proc_prices, n_pips=n_pips, dist_measure=proc_dist_measure)
    pips = []
    for i in range(0,len(pips_x)) :
        pips.append((pips_x[i],in_prices.iloc[pips_x[i]]['Date'],pips_y[i]))
    points = pd.DataFrame(pips,columns=['DateIndex','Date','Price'])
     # 轉換為輸出格式
    perceptually_important_points = points.set_index('DateIndex')
    return perceptually_important_points