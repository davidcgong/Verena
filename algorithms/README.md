# Algorithms

## Bull Sentiment Penny Stock Strategy (PennyStock_Bull_Sentiment.py)

Screens high sentiment penny stocks that are not low float, buys them, holds them overnight, and sells the next day

![Penny Stock Bull Strat. Results](https://i.imgur.com/8azl8Cd.png)

Return - 160% annualized (not accounting for 4000% return overall because of invalid data from Quantopian on that 1 day.)
Sharpe - 4.93
Max Drawdown - -7.18%
Position Concentration - 0.50%
Average Leverage - 0.90x

Mostly focused in Healthcare Sector - to be expected because that is where a lot of big plays are made. Think of BPTH (Bio-path Holdings)

### Problems
1) Data doesn't seem to be correct. Returns are definitely not accurate when logs were viewed (portfolio could not have grown by 1500% in 1 day)
2) Sentiment data only starts from 2014. Cannot measure how strong it would be in the event of a market crash, though penny stocks seem to be in a world of their own
3) Some winners, some losers
4) Needs to be tested live
