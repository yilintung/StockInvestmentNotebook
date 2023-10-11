#
# @file StockRankEvaluate.py
#

import numpy as np

##### 參考 ： https://vocus.cc/article/62ada936fd89780001fe6208

class StockRankEvaluate :
    
    ##### StockRankEvaluate類別的建構子 #####
    # base_prices : 基準價格，一般使用收盤價 (型態為numpy.ndarray)
    def __init__( self, base_prices) :
        if base_prices is not None and type(base_prices) is np.ndarray :
            self.base_prices_mean = np.mean(base_prices)
            self.base_prices_std  = np.std(base_prices)
            # 除錯訊息
            print("基準價格區間 ： 平均值 = {:.2f} ， 標準差 ＝ {:.2f} ， 最高 ＝ {:.2f} ， 最低 ＝ {:.2f}".format(self.base_prices_mean,self.base_prices_std,np.max(base_prices),np.min(base_prices)))
        else :
            raise ValueError
    
    ##### 評價函式 #####
    # price : 待評價之價格 (float)
    def eval( self, price) :
        rank = ''
        price_std_range = (price - self.base_prices_mean) / self.base_prices_std
        if price_std_range <= -2.0 :
            rank = '極低'
        elif price_std_range > -2.0 and price_std_range < -1.0 :
            rank = '低'
        elif price_std_range >= -1.0 and price_std_range <= 1.0 :
            rank = '中'
        elif price_std_range > 1.0 and price_std_range < 2.0 :
            rank = '高'
        elif price_std_range >= 2.0 :
            rank = '極高'
        # 除錯訊息
        print("價格 ＝ {:.2f}元（標準差範圍 ＝ {:.2f}） ， 位階評價：{}".format(price,price_std_range,rank))
        return rank