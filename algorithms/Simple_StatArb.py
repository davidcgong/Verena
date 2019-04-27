# Beta-neutral

import numpy as np
import statsmodels.api as smapi

   
def initialize(context):
    context.HEDGE = sid(8554)
    context.stocks = symbols('AAPL','AAP','ABC','ADS','AGN','ALXN','AMG','AMGN','AMP','AMZN','ANTM','APD','AVB','AZO','BA','BCR','BDX','BIIB','BLK','BMRN','BXP','CB','CELG','CF','CHTR','CI','CLX','CMG','CMI','COST','CVS','CVX','CXO','DIS','ECL','EQIX','ESS','EW','FDX','FLT','GD','GILD','GMCR','GS','GWW','GOOG','HBI','HD','HON','HSIC','HSY','HUM','IBM','ICE','IEP','ILMN','ISRG','IVV','KMB','KSU','LLL','LMT','LNKD','MCK','MHFI','MHK','MJN','MKL','MMM','MNST','MON','MPC','MTB','NEE','NFLX','NOC','NSC','ORLY','PANW','PCLN','PCP','PCYC','PH','PII','PLL','PPG','PSA','PX','PXD','REGN','RL','ROK','ROP','RTN','SBAC','SHW','SIAL','SJM','SLG','SPG','SRCL','SRE','STZ','TDG','TMO','TRV','TRW','TSLA','TWC','UHS','UNH','UNP','UPS','UTX','V','VNO','VRTX','WDC','WHR','WYNN')
    set_benchmark(context.HEDGE)
    schedule_function(myfunc,date_rule=date_rules.every_day(),time_rule=time_rules.market_close(minutes=30))
    
def handle_data(context, data):
    record(Leverage=context.account.leverage)
    pass

def myfunc(context, data):
    if get_open_orders():
        return
    prices = history(20, "1d", "price")
    prices = prices.dropna(axis=1)
    prices = prices.drop([context.HEDGE], axis=1)
    ret = prices.pct_change(5).dropna()
    ret = np.log1p(ret).values
    cumret = ret #np.cumsum(ret, axis=0)
    hedge = np.mean(cumret, axis=1)
        
    i = 0
    score = []
    for sid in prices:
        diff = np.diff(cumret[:,i])
        X = smapi.add_constant(diff, prepend=True)
        Y = np.diff(cumret[:,i] - hedge)
        res = smapi.OLS(Y, X).fit()
        if len(res.params) > 1:
            score.append(res.params[1])
        else:
            score.append(0)
        i += 1
        
    netscore = np.sum(np.abs(score))
    
    i = 0
    wsum = 0
    for sid in prices:
        try:
            # val = 500000 * score[i] / netscore
            val = context.portfolio.portfolio_value * score[i] / netscore
            order_target_value(sid,  val)
            wsum += val
        except:
            log.info("exception")
            i += 1
            continue
            
        i += 1      
    order_target_value(context.HEDGE, -wsum)