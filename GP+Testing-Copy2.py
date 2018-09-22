#GP requirements (+zipline)
import random
import operator

import deap
import pandas as pd
from deap import gp,base, benchmarks,cma,creator,tools, algorithms
from datetime import datetime
from zipline import run_algorithm
from zipline.api import order, record, symbol, order_target,order_target_percent, schedule_function, date_rules, time_rules
import pytz
import numpy as np
import matplotlib.pyplot as plt
import pyfolio as pf

import uuid


    
def evolveAssetStrategy(asset,cashAmount=5000,priceDataRange=30, minTrades=50,
                        maxDepth=15,popSize=20,cxpbg=0.5,mprobg=0.1,ngens=10,hofSize=3,
                        trainStart=datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc),
                        trainEnd=datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc),
                        testStart=datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc),
                        testEnd=datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc),
                        riskAdjustment=0):
# RiskAdjustment setting .. if 0, MDD is not considerd. If 1, semi-conservative objective function, If 2, most conservative objective function
    

        ## set model parameters
        # set cash to trade, leverage limits
        selectedAsset=asset
        cash=cashAmount
        tStart=trainStart
        tEnd=trainEnd
        teStart=testStart
        teEnd=testEnd
        RA=riskAdjustment
        # set number of time steps to include in decision rules
        argcount=priceDataRange
        maxStrategyDepth=maxDepth
        tradeThresh=minTrades
        
        ## custom tree functions
        print('defining functions')
        def orderasset(quantity):
            order(selectedAsset,quantity)
        def if_then_else_float(input, output1, output2):
            return output1 if input else output2
        def if_then_else_bool(input,output1,output2):
            return output1 if input else output2
        def if_then_else_comb(input,output1,output2):
            return output1 if input else output2
        def protectedDiv(left, right):
            try: return left / right
            except ZeroDivisionError: return 1
        
        
        print('defining sets')
        # define the function and terminal sets
        
        pset = gp.PrimitiveSet("main",argcount)
        #pset.addPrimitive(operator.xor,2)
        pset.addPrimitive(operator.mul,2)
        #pset.addPrimitive(if_then_else_float,3)
        pset.addTerminal(1)
        #pset.addPrimitive(operator.and_,2)
        #pset.addPrimitive(operator.or_,2)
        #pset.addPrimitive(operator.not_,1)
        pset.addPrimitive(operator.add,2)
        pset.addPrimitive(operator.sub,2)
       # pset.addPrimitive(operator.ge,2)
        #pset.addPrimitive(operator.le,2)
        #pset.addPrimitive(if_then_else_float,3)
        pset.addPrimitive(protectedDiv,2)
        i=0
        while i<=5:
            j=np.random.uniform(-1,10)
            pset.addTerminal(j)
            i+=1
        i=0
        
        # DESIRED CHANGE ++> probabilistically select fewer integers from same range to increase density of price data in strategy
        while i<=5:
            j=np.random.uniform(-1,1)
            pset.addTerminal(j)
            i+=1
    # instantiate multiprocessing pool
    
        #pool=multiprocessing.Pool()
    
    # create individuals
        
        expr = gp.genGrow(pset, min_=1, max_=maxStrategyDepth)
        tree = gp.PrimitiveTree(expr)
        print(tree)
        # creator
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        # toolbox
        
        toolbox = base.Toolbox()
        #toolbox.register("map",pool.map)
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=maxStrategyDepth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
     
        # define function such that it takes a tree individual as input 
        def maxProfit(individual):
                profit=-1
                tradingRule = toolbox.compile(expr=individual)
                
                def initialize(context):
                    print('init maxprofit')
                    context.dayCount = 0
                    context.tradeCount=0
                    context.daily_message = "Day {}."
                    context.weekly_message = "Time to place some trades!"
                    context.asset = symbol(selectedAsset)
                    context.strategyDescription=pd.DataFrame()
                    # clear portfolio  15 min prior to close
                    #context.schedule_function(zeroPortfolio,date_rules.every_day(),time_rules.market_close(minutes=30))
        
                def handle_data(context, data):
                    #print('handle data')
                    prices = data.history(context.asset, 'price',argcount,'1d') 
                    normalizedPrices=(np.divide(prices,np.mean(prices)))
                    scaledPrice=np.divide((normalizedPrices-np.min(normalizedPrices)),(np.max(normalizedPrices)-np.min(normalizedPrices)))
                    dp=tradingRule(*scaledPrice)#[0],scaledPrice[1],scaledPrice[2],scaledPrice[3],scaledPrice[4],scaledPrice[5])
                    #print(dp) 
                 #   dpList.append(dp)
                    if dp<0:
                        desiredPosition=max(-1,dp)
                    else:
                        desiredPosition=min(1,dp)
                    #print('day count')
                   # print(context.dayCount)

                    # print(desiredPosition)
                    # if desired position varies from previous desired position by more than 10%, order to target percentage 
                    currentPosition=np.divide((context.portfolio.positions[context.asset].amount)*(context.portfolio.positions[context.asset].cost_basis),context.portfolio.portfolio_value)
                    if np.abs(desiredPosition-currentPosition)>0.1:
                        order_target_percent(context.asset,desiredPosition)
                        context.tradeCount+=1

                    context.dayCount += 1
                    
 
            
                capital_base = cash
                start = tStart
                end = tEnd
                #validate = datetime(2018,1,1,0,0,0,0,pytz.utc)
                results=run_algorithm(start = start, end = end, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl')
                #validationSet=run_algorithm(start = end, end = validate, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl')
                TL=results['transactions'].tolist()
                transactionList=[x for x in TL if x]
                
                AVL=results['algo_volatility'].tolist()
                algo_volatility=AVL[-1]
                #print(transactionList)
                #print(nonEmptyTransactions.to_frame)
                transactionCount=len(transactionList)
                #print(transactionCount)
                profits=results['pnl']
                drawdown=results['max_drawdown'].tolist()
                #print(drawdown)
                #print(profits)
                if RA==0:
                    if transactionCount>=tradeThresh:
                        profit=np.divide(np.sum(profits),capital_base)
                if RA==1:
                    if transactionCount>=tradeThresh:
                        profit=np.divide(np.sum(profits),capital_base)+drawdown[-1]
                if RA==2:
                    if transactionCount>=tradeThresh:
                        if drawdown[-1]!=0:
                            profit=np.divide(np.divide(np.sum(profits),capital_base),-1*drawdown[-1])
                return profit,
    
        
        
        #print('printing final portfolio value')
        #print(p_value)
        
    
        
        ## register evaluation, selection, crossover, and mutation functions
        print('toolbox reg')
        print("register evaluation")
        toolbox.register("evaluate", maxProfit)
        print("register selection")
        toolbox.register("select", tools.selTournament, tournsize=3)
        print("register mate")
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxStrategyDepth))
        toolbox.decorate("expr_mut", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxStrategyDepth))
        print("register statistics")
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        def trainTestFitness(trainScore,testScore):
            train=np.exp(-1*trainScore)
            test=np.exp(-1*testScore)
            score=np.abs(train-test)+1-np.divide(np.sqrt((1-train)**2+(1-test)**2),np.sqrt(2))
            if((train<0 or train>1)):
                score=-1
            if ((test<0) or (test>1)):
                score=-1
            return score
        #def generateTearSheet(results):
            
        # define function to simulate and illustrate the hall of fame individuals
        print("defining evaluateWinners")
        def evaluateWinners(HOF):
            HOFList=[]
            for i in range(0, len(HOF)):
                threshold=tradeThresh
                
                tradingRule = toolbox.compile(expr=HOF[i])
                #create lists to track portfolio
                #desiredPositionList=[]
                #positionList=[]
                #cashList=[]
                #pnlList=[]
                #dpList=[]
                #tradeCountList=[]
                
                def testEvaluation(individual,fileNameString):
                    fileID=str(uuid.uuid4())
                    def initialize(context):
                        #print('init maxprofit')
                        profit=-1     
                        context.dayCount = 0
                        context.tradeCount=0
                        context.daily_message = "Day {}."
                        context.weekly_message = "Time to place some trades!"
                        context.asset = symbol(selectedAsset)
                        context.strategyDescription=pd.DataFrame()
                        # clear portfolio  15 min prior to close
                        #context.schedule_function(zeroPortfolio,date_rules.every_day(),time_rules.market_close(minutes=30))
            
                    def handle_data(context, data):
                        #print('handle data')
                        prices = data.history(context.asset, 'price',argcount,'1d') 
                        normalizedPrices=(np.divide(prices,np.mean(prices)))
                        scaledPrice=np.divide((normalizedPrices-np.min(normalizedPrices)),(np.max(normalizedPrices)-np.min(normalizedPrices)))
                        dp=tradingRule(*scaledPrice)#[0],scaledPrice[1],scaledPrice[2],scaledPrice[3],scaledPrice[4],scaledPrice[5])
                      #  print(dp) 
                     #   dpList.append(dp)
                        if dp<0:
                            desiredPosition=max(-1,dp)
                        else:
                            desiredPosition=min(1,dp)

                        # if desired position varies from previous desired position by more than 10%, order to target percentage 
                        currentPosition=np.divide((context.portfolio.positions[context.asset].amount)*(context.portfolio.positions[context.asset].cost_basis),context.portfolio.portfolio_value)
                        if np.abs(desiredPosition-currentPosition)>0.1:
                            order_target_percent(context.asset,desiredPosition)
                            context.tradeCount+=1
                        
                        context.dayCount += 1
                        
                        record(Asset=data.current(context.asset, 'price'))
                        

                        
                    def analyzeTrain(context, perf):
                        print('analyseTrain')
                        fig = plt.figure()
                        #plt.title('Individual Performance Characteristics - Training Set')
                        ax1 = fig.add_subplot(211)
                        perf.portfolio_value.plot(ax=ax1)
                        ax1.set_ylabel('portfolio value in $')
                        ax1.set_title('Portfolio Value | ' + selectedAsset+" | Training Set")
                        ax2 = fig.add_subplot(212)
                        perf['Asset'].plot(ax=ax2)
                        #perf[['short_mavg', 'long_mavg']].plot(ax=ax2)
                    
                        perf_trans = perf.ix[[t != [] for t in perf.transactions]]
                        buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
                        sells = perf_trans.ix[
                            [t[0]['amount'] < 0 for t in perf_trans.transactions]]
                        ax2.plot(buys.index, perf.Asset.ix[buys.index],
                                 '^', markersize=5, color='m')
                        ax2.plot(sells.index, perf.Asset.ix[sells.index],
                                 'v', markersize=5, color='k')
                        ax2.set_ylabel('price in $')
                        ax2.set_title('Trade Activity | ' + selectedAsset+" | Training Set")
                        plt.legend(['Price','Buy','Sell'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                       
                        
                        plt.tight_layout()
                        plt.show()
                    def analyzeTest(context, perf):
                        print('analyzetest')
                        fig = plt.figure()
                        #plt.title('Individual Performance Characteristics - Test Set')
                        ax1 = fig.add_subplot(211)
                        perf.portfolio_value.plot(ax=ax1)
                        ax1.set_ylabel('portfolio value in $')
                        ax1.set_title('Portfolio Value | ' + selectedAsset+" | Test Set")
                        ax2 = fig.add_subplot(212)
                        perf['Asset'].plot(ax=ax2)
                        #perf[['short_mavg', 'long_mavg']].plot(ax=ax2)
                    
                        perf_trans = perf.ix[[t != [] for t in perf.transactions]]
                        buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
                        sells = perf_trans.ix[
                            [t[0]['amount'] < 0 for t in perf_trans.transactions]]
                        ax2.plot(buys.index, perf.Asset.ix[buys.index],
                                 '^', markersize=5, color='m')
                        ax2.plot(sells.index, perf.Asset.ix[sells.index],
                                 'v', markersize=5, color='k')
                        ax2.set_ylabel('price in $')
                        ax2.set_title('Trade Activity | ' + selectedAsset+" | Test Set")
                        plt.legend(['Price','Buy','Sell'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                        #plt.legend(['Price','Buy','Sell'],loc=0)
                        plt.tight_layout()
                        plt.show()
                    
                
                    
                        
                    capital_base = cash
                    trainProfit=-1
                    testProfit=-1
                    ## CAPTURE TRAINING RESULTS
                    start = tStart
                    end = tEnd
                    #validate = datetime(2018,1,1,0,0,0,0,pytz.utc)
                    trainResults=run_algorithm(start = start, end = end, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl', analyze=analyzeTrain)
                    #validationSet=run_algorithm(start = end, end = validate, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl')
                    TL=trainResults['transactions'].tolist()
                    transactionList=[x for x in TL if x]
                    #print(transactionList)
                    #print(nonEmptyTransactions.to_frame)
                    transactionCount=len(transactionList)
                    #print(transactionCount)
                    profits=trainResults['pnl']
                    if transactionCount>=threshold:
                        trainProfit=np.divide(np.sum(profits),capital_base)
                    ## CAPTURE TESTING RESULTS
                    
                    start = teStart
                    end = teEnd
                    #validate = datetime(2018,1,1,0,0,0,0,pytz.utc)
                    testResults=run_algorithm(start = start, end = end, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl',analyze=analyzeTest)
                    #validationSet=run_algorithm(start = end, end = validate, initialize=initialize,                capital_base=capital_base, handle_data=handle_data,                bundle = 'quantopian-quandl')
                    TL=testResults['transactions'].tolist()
                    transactionList=[x for x in TL if x]
                    #print(transactionList)
                    #print(nonEmptyTransactions.to_frame)
                    transactionCount=len(transactionList)
                    #print(transactionCount)
                    profits=testResults['pnl']
                    if transactionCount>=threshold:
                        testProfit=np.divide(np.sum(profits),capital_base)
                    fitness=trainTestFitness(trainProfit,testProfit)
                    return trainProfit,testProfit,fitness
                
                
                HOFList.append(HOF[i])
                HOFList.append(testEvaluation(HOF[i],str(i)))
            return HOFList
        
        
        #%% run the evolution 
        def runEvolution(population,cxpb,mprob,generations,recordNum):
            pop = toolbox.population(n=population)
            hof = tools.HallOfFame(recordNum)
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mprob, generations, stats=mstats,
                                           halloffame=hof, verbose=True)
            hof_test_scores= evaluateWinners(hof)
            return hof,log,hof_test_scores 
        hallOfFame,logg,testScores=runEvolution(popSize,cxpbg,mprobg,ngens,hofSize)
        
        for i in range(0,len(testScores)):
            print('winner' + str(i))
            #print(hallOfFame[i])
            print(testScores[i])
        #print(logg)
        gen=logg.select("gen")
        avgFit, maxFit = logg.chapters['fitness'].select("avg", "max")
        print(gen)
        print(avgFit)
        fig2=plt.figure()
        plt.plot(gen,avgFit)
        plt.plot(gen,maxFit)
        plt.ylim(ymin=0)
        plt.title('fitness vs. generations')
        plt.legend(['Average Training Fitness','Max Training Fitness'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Cumulative Return (Training)')
        plt.xlabel('Generation')
        return fig2, testScores
    
        
        
#%% Test the function call 

assetList=['FB']
    
for asset in assetList:
    figure, performance=evolveAssetStrategy(asset,cashAmount=10000, minTrades=30,
                                maxDepth=15,popSize=5,cxpbg=0.5,mprobg=0.1,ngens=5,hofSize=20,
                                trainStart=datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc),
                                trainEnd=datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc),
                                testStart=datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc),
                                testEnd=datetime(2018, 1, 1, 0, 0, 0, 0, pytz.utc),priceDataRange=30,
                                riskAdjustment=2)
    figure.show()
    print(performance)
    np.savetxt('performance.txt',performance,delimiter='')
    
