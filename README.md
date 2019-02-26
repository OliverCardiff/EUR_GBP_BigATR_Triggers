# EUR_GBP_BigATR_Triggers
Development of trading strategy for GBP/USD using partial EUR/USD correlations as predictors

This project eschewed the use of 'backtrader' in this case, as increased efficiency was required for optimisation loops. Instead a smaller custom trade simulator was built to iterate over an arbitrary timeframe, with all trades simulated on minute-level data.
 
## Mission statement 
The objective of this task is to build models which yield a profitable set of trades for the GBP/USD FX pair, informed by the movements of the EUR/USD pair. The mantra ‘simple models work best’ will be adhered to. The strategy will focus on the detection of larger moves in EUR/USD, and the proportional relationship between the equivalent move in GBP/USD at a simultaneous timestep. Statistical learning models will be used to predict price maxima and minima within a time window t of a trigger x. The ideal model will output a simple estimate of price, with sufficient predictive power to inform a trading strategy with an edge. 

## Simulation of Commissions
Commissions are assumed to be taken from a broker which widens spread (like Oanda), a conservatively high 2-pip commission per trade was deducted from the portfolio.

## Use of Tech/Data 
• Language: Python 3.5 

• Scikit-learn, for GridSearchCV and modelling functions 

• TA-lib python wrapper, for the technical indicator functions 

• Bayesian-optimisation library, for optimising black-box functions 

• Pandas and Numpy for data manipulation 

• Matplotlib for graph generation 

• Free FX minute data from 2001-2018 retrieved from: https://forextester.com/data/datasources 

## Model Use 

Random Forest Regression models from scikit-learn were used to estimate prices. GridSearchCV was used to tune hyperparameters, with 4-fold time-slice CV. Most often the grid search ended up choosing small numbers of features in decisions per split, and shallow tree depths. The modal minimum-MSE tree found during trigger optimisation (which was ultimately abandoned) had a depth of 4, with 3 random features considered per node split, and ~50 estimators in the ensemble. The final optimised estimator parameter sets are quite like this also (see below) 
Another model type which could be tried is Bayesian-ridge regression. The useful thing about BR is that it treats all features as gamma PDFs, given learned probabilities, and the regression fit uses these to generate a standard deviation for each of its estimates. This may allow a more adaptive tuning of stops, given information about the variance of price estimates. This does come with more feature pre-processing requirements than RF, however. 


## Predictive Features 
```
ATR-Relative trigger scale: abs(openEUR – closeEUR) / ATR(20) 
Immediate GBP dollar-proportional move: (dEUR / openEUR) / (dGBP / openGBP) at t0 
Log(closeGBP/MASlow(GBP)) at t0  (log to preserve distribution symmetry) 
Log(closeGBP/MAFast(GBP)) at t0 
Log(closeEUR/MASlow(EUR)) at t0 
Log(closeEUR/MAFast(EUR)) at t0 
RSI(EURUSD) at t0 
RSI(GBPUSD) at t0 
ATR(EURUSD) at t0 
ATR(GBPUSD) at t0 
##More to be added or removed 
```
![Correlation_of_preds](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/multi_corr.png)

The fast SMA was removed due to it's high colinearity with other predictors. 

The free data availability for volume was not of enough quantity or quality compared to the price data, so volume predictors have not been included. 

Predicted Features Where x is the trigger index, and t is the timestep count. 

Maximum price in time window. Max( highGBP [ x: t+x ] ) 

Minimum price in time window. Min( lowGBP [ x: t+x ] ) 

 
## Modelling outcomes for optimised RFRs

![hyperparams](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/mini_tab.png)
*Optimised hyperparameters for each RFR*
![feature_importance](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/Model_features.png)
*Feature Importances extracted from CV best estimators*

It is remarkable that the features differ in importance so much by just changing the timeframe. Although the ATR of GBP does seem to be a regular feature. 

## Optimisation Loops 

Two applications of the Bayesian ‘black-box’ function optimisation where attempted.  The first was an optimisation of the trade entry/signal generation function. The parameters were the ATR scaling factor and the time window before stop. The opt function sought out the pair of price prediction models with the minimum combined MSE. The objective of optimising this function was to find a signal set which gave more predictive power to the model building process, and ergo which could be more reliably trained. 

![trigger_loops](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/trigger_opt.png)

*This is an outline of the trigger optimisation*

The other application of the Bayesian optimiser was to tune the parameters of the trading system that used the predictive models. The optimiser sought to maximise returns (naturally) and tuned buy and sell stop adjustments, the risk-ratio, and ATR scaling factor again. In the end the risk-ratio was set at 1.2 and excluded from tuning though. 
Both optimisation loops were performed on the same training data. The test data (60% of the entire time series) was left aside for testing at the end, this is shown in some graphs below. 

 ![train_opts](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/trade_opt.png)
 
*This is an outline of the optimisation of the trade system*

## Results
I spent a huge amount of time just figuring out which bits should or should not be optimised, which parameters to expose at which range etc. There are also a few erroneous bits of the timeseries data near the train/test split. I adjusted the split window a little bit so it would be excluded (otherwise the optimiser would just seek that out and exploit it at all costs).
 
![4hgraph](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/4H_Train_Test.png)

*Result of model building on the 4H timeframe*

Core parameters:

• ATR scaling factor: 0.8 (RF training),0.72 (System optimisation)  

• Qloss: 0.915 

• QProf: 0.93 

• Risk_rate = 1.2 

• Trade_window = 6 (6 * 4H = 24H!) 

This model will trade GBP/USD over the course of up to 24hrs based on the observation of a single 4hr move greater than 80% of the ATR20, in the EUR/USD pair.  

![1dgraph](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/1D_Train_Test.png)

*Result of model building on the 1D timeframe*

Core parameters:

• ATR scaling factor: 0.8 (RF training),0.79 (System optimisation)  

• Qloss: 1.175 

• QProf: 0.946 

• Risk_rate = 1.2 

• Trade_window = 1 (1D = 24H!) 

This model simply trades GBP/USD over a 24hr window based on a large move having occurred on the previous day in EUR/USD. 
These two models worked well. Many models failed for many reasons, many of which were wonky optimisations. I found that the test models became profitable (& more robust to permutations) almost immediately after I made the decision to optimise less and fix the initial ATR scale at 0.8, fix the Risk-ratio at 1.2, and fix the time window at 24Hr for both. Simpler models work best!  

## Notes 

Optimising the trigger function didn’t really work. The issue was that minimising error by scale always pushed the time window to the minimum value, and ATR scale had minimal effect on it. The value of the scaling factor was best guessed at by hand. 
Optimising the risk-ratio worked sometimes, but often led to problems. For example, the optimiser would often crank it up until nearly no trades were performed, then tweak the other params until profitable ones started popping above the threshold. Performing 4-5 1-day trades over 10 years didn’t seem like a great idea, so this was scrapped too, 1.2 seems to work okay. 
One-hour time-frame trading has yet to be attempted (not enough time right now to compute as the opt function takes 4x as long as the 4H).

![1hgraph](https://github.com/OliverCardiff/EUR_GBP_BigATR_Triggers/blob/master/1H_Train_Test.png)

*This is the best 1-hr system (Scrapped!)*


