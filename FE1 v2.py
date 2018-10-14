import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
%matplotlib inline 

rets = pd.read_csv('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project\Selected_Industry_Returns.csv')
del rets['Month']
rets120 = rets.copy()
rets120 = rets120[:120]
rets120 = rets120
rets120

weights = [0.2031850348,0.2214003473,0.1110603117,0.1290694581,-0.04089010302,-0.08342529211,0.2622203278, 0.1107894742, -0.1678231688, 0.2544136101]
weights = np.array(weights)
rbar_in = [0.75892,0.85667,0.88617,0.61467,0.39033, 1.03908, 0.97250, 0.96925, 1.68158, 0.47542]

#optimise
def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(rbar_in* weights) * 12
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets120.cov() *12, weights)))
    return np.array([pret, pvol, (pret-3.549)/pvol]) # assumes rf=2.4%, need to fix, 3rd entry returns sharpe ratio
 
import scipy.optimize as sco

def min_sharpe(weights):
    return -statistics(weights)[2]
    
cons=({"type":"eq","fun":lambda	x:np.sum(x)-1}) #all parameters sum to 1
bnds=tuple((-1,	1)for x	in range(10)) #weights are betwwen -100% and 100%

optimised = sco.minimize(min_sharpe, 10 * [1./10], method = 'SLSQP', bounds = bnds, constraints = cons)
optimised
optimised['x'].round(3)

statistics(optimised['x']).round(3)

#minimise variance
def min_var(weights): 
    return statistics(weights)[1]**2
    
optvar = sco.minimize(min_var, 10 * [1./10],method = 'SLSQP', bounds=bnds, constraints = cons)
optvar
optvar['x'].round(3)
statistics(optvar['x']).round(3)

#efficient frontier
cons = ({'type': 'eq', 'fun':lambda x: statistics(x)[0] - tret},{'type':'eq','fun':lambda x: np.sum(x)-1})
bnds = tuple((-1,1) for x in weights)

def min_port(weights):
    return statistics(weights)[1]
    
trets = np.linspace(0,35,70)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun':lambda x: statistics(x)[0] - tret},{'type':'eq','fun':lambda x: np.sum(x)-1})
    res = sco.minimize(min_port, 10 * [1./10], method = 'SLSQP', bounds = bnds, constraints = cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)


#capital market line
import scipy.interpolate as sci
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)

def f(x):
    return sci.splev(x, tck, der=0)
    
def df(x):
    return sci.splev(x, tck, der=1)
    
def equations(p, rf=3.549):
    eq1 = rf - p[0]
    eq2 = rf +p[1]*p[2]-f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3
    
opt = abs(sco.fsolve(equations, [3.549, 50, 15]))
opt # first entry should be risk free rate

np.round(equations(opt), 9)#want equations = 0

cx = np.linspace(0.0, 40)
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0]-f(opt[2])},{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
res = sco.minimize(min_port, 10*[1./10], method = 'SLSQP', bounds = bnds, constraints=cons)
res['x'].round(6)
tangent_weights = res['x']


plt.figure(figsize=(12,6))
plt.scatter(tvols, trets, c= 'maroon', marker='x')
plt.scatter(rets120['Food '].std()*np.sqrt(12), rbar_in[0]*12, c= (rbar_in[0]*12-3.549)/rets120['Food '].std()*np.sqrt(12), marker ='o')
plt.annotate('Food',xy=(rets120['Food '].std()*np.sqrt(12),rbar_in[0]*12))
plt.scatter(rets120['Hshld'].std()*np.sqrt(12), rbar_in[1]*12, c= (rbar_in[1]*12-3.549)/rets120['Hshld'].std()*np.sqrt(12), marker ='o')
plt.annotate('Hshld',xy=(rets120['Hshld'].std()*np.sqrt(12),rbar_in[1]*12))
plt.scatter(rets120['Drugs'].std()*np.sqrt(12), rbar_in[2]*12, c= (rbar_in[2]*12-3.549)/rets120['Drugs'].std()*np.sqrt(12), marker ='o')
plt.annotate('Drugs',xy=(rets120['Drugs'].std()*np.sqrt(12),rbar_in[2]*12))
plt.scatter(rets120['Chems'].std()*np.sqrt(12), rbar_in[3]*12, c= (rbar_in[3]-3.549)/rets120['Chems'].std()*np.sqrt(12), marker ='o')
plt.annotate('Chems',xy=(rets120['Chems'].std()*np.sqrt(12),rbar_in[3]*12))
plt.scatter(rets120['Txtls'].std()*np.sqrt(12), rbar_in[4]*12, c= (rbar_in[4]*12-3.549)/rets120['Txtls'].std()*np.sqrt(12), marker ='o')
plt.annotate('Txtls',xy=(rets120['Txtls'].std()*np.sqrt(12),rbar_in[4]*12))
plt.scatter(rets120['Aero '].std()*np.sqrt(12), rbar_in[5]*12, c= (rbar_in[5]*12-3.549)/rets120['Aero '].std()*np.sqrt(12), marker ='o')
plt.annotate('Aero ',xy=(rets120['Aero '].std()*np.sqrt(12),rbar_in[5]*12))
plt.scatter(rets120['Util '].std()*np.sqrt(12), rbar_in[6]*12, c= (rbar_in[6]*12-3.549)/rets120['Util '].std()*np.sqrt(12), marker ='o')
plt.annotate('Util ',xy=(rets120['Util '].std()*np.sqrt(12),rbar_in[6]*12))
plt.scatter(rets120['Softw'].std()*np.sqrt(12), rbar_in[7]*12, c= (rbar_in[7]*12-3.549)/rets120['Softw'].std()*np.sqrt(12), marker ='o')
plt.annotate('Softw',xy=(rets120['Softw'].std()*np.sqrt(12),rbar_in[7]*12))
plt.scatter(rets120['Fin  '].std()*np.sqrt(12), rbar_in[8]*12, c= (rbar_in[8]*12-3.549)/rets120['Fin  '].std()*np.sqrt(12), marker ='o')
plt.annotate('Fin  ',xy=(rets120['Fin  '].std()*np.sqrt(12),rbar_in[8]*12))
plt.scatter(rets120['RlEst'].std()*np.sqrt(12), rbar_in[9]*12, c= (rbar_in[9]*12-3.549)/rets120['RlEst'].std()*np.sqrt(12), marker ='o')
plt.annotate('RlEst',xy=(rets120['RlEst'].std()*np.sqrt(12),rbar_in[9]*12))
plt.plot(statistics(optimised['x'])[1],statistics(optimised['x'])[0], 'g*', markersize=15.0,label = 'Tangent Portfolio')#highest sharpe ratio - green star
plt.plot(statistics(optvar['x'])[1],statistics(optvar['x'])[0],'r*',markersize=15.0,label='GMVP')#GMVP - red star
plt.grid(True)									
plt.xlabel('Expected Volatility')									
plt.ylabel('Expected Return')									
plt.title('In-Sample Efficient Frontier')
plt.plot(cx, opt[0] + opt[1]*cx, lw=1.5,label='Capital Allocation Line')
plt.legend(loc='upper left',numpoints=1)
plt.axis([0, 40, 0, 40])
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/in_sample_frontier.jpeg',bbox_inches='tight')

						

#OUT Of SAMPLE
#-------------
tangent_weights = np.array(tangent_weights)
GMVP_weights = weights
GMVP_weights = np.array(GMVP_weights)
rets240 = rets.copy()
rets240 = rets240[-120:]

###HINDSIGHT
g = rets240*GMVP_weights*12
g = g.mean(axis=1)
g = np.array(g)


t = rets240*tangent_weights*12
t = t.mean(axis=1)
t = np.array(t)


plt.figure(figsize=(12,4))
plt.plot(g,label = 'GMVP',lw=1.5)
plt.plot(t,label='Tangent',lw=1.5)
plt.grid(True)
plt.title('Out of Sample Hindsight Efficient Returns')
plt.legend(loc='upper left')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Hindsight Time Series.jpeg',bbox_inches='tight')

hindsight_gmvp_cum = []
for i in range(len(g)-1):
    if i == 0:
        hindsight_gmvp_cum.append((g[i])*(1+(g[i+1]/100)))
    else:
        if hindsight_gmvp_cum[i-1] > 0:
            hindsight_gmvp_cum.append(hindsight_gmvp_cum[i-1]*(1+(g[i]/100)))
        else:
            hindsight_gmvp_cum.append(hindsight_gmvp_cum[i-1]+((g[i]/100)))
              

hindsight_tan_cum= []
for i in range(len(t)-1):
    if i == 0:
        hindsight_tan_cum.append((t[i])*(1+(t[i+1]/100)))
    else:
        if hindsight_tan_cum[i-1] > 0:
            hindsight_tan_cum.append(hindsight_tan_cum[i-1]*(1+(t[i]/100)))
        else:
            hindsight_tan_cum.append(hindsight_tan_cum[i-1]+((t[i]/100)))      

out_data= [3.64,2.13,1.27,1.84,-1.58,1.11,3.93,3.65,-1.56,-3.33,1.34,3.54,2.12,-4.49,-0.6,-6.15,-2.96,-0.76,4.78,2.04,-8.27,-0.62,1.66,-9.09,-17.15,-7.83,1.74,-8.12,-10.09,8.97,10.2,5.21,0.44,7.73,3.34,4.09,-2.59,5.56,2.76,-3.36,3.4,6.32,2.01,-7.88,-5.55,6.94,-4.76,9.55,3.89,0.61,6.83,2,3.5,0.46,2.9,-1.27,-1.75,-2.36,-5.98,-7.59,11.35,-0.28,0.74,5.05,4.42,3.11,-0.85,-6.18,3.89,0.79,2.56,2.74,-1.75,0.79,1.19,5.57,1.29,4.03,1.56,2.8,-1.2,5.65,-2.71,3.77,4.18,3.12,2.81,-3.32,4.65,0.43,-0.19,2.06,2.61,-2.04,4.24,-1.97,2.52,2.55,-0.06,-3.11,6.13,-1.12,0.59,1.36,-1.53,1.54,-6.04,-3.07,7.75,0.56,-2.16,-5.76,-0.05,6.98,0.92,1.79,-0.02,3.97,0.51,0.27]

out_market_cum = []
for i in range(119):
    if i == 0:
        out_market_cum.append((out_data[i])*(1+(out_data[i+1]/100)))
    else:
        if out_market_cum[i-1] > 0:
            out_market_cum.append(out_market_cum[i-1]*(1+(out_data[i]/100)))
        else:
            out_market_cum.append(out_market_cum[i-1]+((out_data[i]/100)))
        
#plot cumulative returns
pylab.figure(figsize=(12,6))
pylab.plot(hindsight_gmvp_cum, label='GMVP',lw=2)
pylab.plot(hindsight_tan_cum,label = 'Tangent',lw=2)
pylab.plot(out_market_cum,label = 'Market',lw=2)
pylab.xlabel('Month')
pylab.ylabel('Cumulative Monthly Returns')
pylab.grid(True)
pylab.title('Cumulative Hindsight Efficient Returns')
pylab.legend(loc = 'upper left')
pylab.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Hindsight Cumulative Returns.jpeg')




#Out of sample returns for GMVP
out_returns_gmvp= []
cov_update = []
Wgmvp_update = []
Wgmvp_update.append(GMVP_weights)
for i in range(len(rets240)):
    out_returns_gmvp.append(Wgmvp_update[i].dot(rets240.iloc[i]))
    rets120.append(rets240.iloc[i])
    cov_update = DataFrame.cov(rets120)
    C_update = dot(np.ones(10).dot(inv(cov_update)),np.ones(10))
    Wgmvp_update.append((1/C_update)*(inv(cov_update).dot(np.ones(10))))
   
 
#Out of sample returns for Tangent Portfolio
rbar = [0.951916667, 0.693083333,0.95575,1.098083333,1.461083333,0.930833333,0.725,1.05025,0.3695,0.531833333]
rbar = np.array(rbar)*100
rets = rets120.copy() #resets for updating
out_returns_tan = []
cov_update = [] #resets for updating
Wtan_update = []
Wtan_update.append(tangent_weights)
for i in range(len(rets240)):
    out_returns_tan.append(Wtan_update[i].dot(rets240.iloc[i]))
    rets.append(rets240.iloc[i])
    cov_update = DataFrame.cov(rets)
    A_update = np.dot(rbar.dot(inv(cov_update)), np.ones(10))
    B_update = np.dot(rbar.dot(inv(cov_update)), rbar)
    C_update = dot(np.ones(10).dot(inv(cov_update)),np.ones(10)) #needs to be updated in the loop
    D_update = B_update*C_update - np.power(A_update, 2)
    sub_1_update = np.array([[C_update, -A_update], [-A_update, B_update]])
    sub_2_update = np.dot(inv(cov_update), np.transpose([rbar, np.ones(10)]))
    sub_3_update = sub_1_update.dot(np.transpose([6, 1]))
    Wtan_update.append((1/D_update)*(sub_2_update.dot(sub_3_update)))

#Plots two return timeseries
pylab.figure(figsize=(12,6))
pylab.plot(out_returns_gmvp, label = 'GMVP',lw=1.5)
pylab.plot(out_returns_tan, label = 'Tangent',lw=1.5)
pylab.legend(loc = 'upper right')
plt.grid(True)
pylab.title('Time Series of Monthly Realised Returns With updated weights')
pylab.xlabel('Month')
pylab.ylabel('Return')
pylab.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Out of Sample Time Series.jpeg',bbox_inches='tight')

#####################cumulative returns
#market_rets_out = pd.read_csv('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project\Market Data.csv')
#del market_rets_out['Month']
#market_rets_out = market_rets_out[-120:]
#mark = market_rets_out/100

#mark = mark
#plt.plot(mark)
#cum_gmvp = []
#cum_tang = []
#cum_mark = []

#cum_mark = np.cumprod(mark+1, axis = 0)
#cum_gmvp = np.cumprod(g+1, axis = 0)
#cum_tang = np.cumprod(t+1, axis = 0)

#cum_mark = cum_mark -1
#cum_gmvp = cum_gmvp -1
#cum_tang = cum_tang -1
#plt.figure(figsize=(12,4))
#plt.plot(cum_gmvp)
#plt.plot(cum_tang)
#plt.plot(cum_mark)

#Cumulative return
out_returns_gmvp_cum = []
for i in range(len(out_returns_gmvp)-1):
    if i == 0:
        out_returns_gmvp_cum.append((out_returns_gmvp[i])*(1+(out_returns_gmvp[i+1]/100)))
    else:
        if out_returns_gmvp_cum[i-1] > 0:
            out_returns_gmvp_cum.append(out_returns_gmvp_cum[i-1]*(1+(out_returns_gmvp[i]/100)))
        else:
            out_returns_gmvp_cum.append(out_returns_gmvp_cum[i-1]+((out_returns_gmvp[i]/100)))
              

out_returns_tan_cum= []
for i in range(len(out_returns_tan)-1):
    if i == 0:
        out_returns_tan_cum.append((out_returns_tan[i])*(1+(out_returns_tan[i+1]/100)))
    else:
        if out_returns_tan_cum[i-1] > 0:
            out_returns_tan_cum.append(out_returns_tan_cum[i-1]*(1+(out_returns_tan[i]/100)))
        else:
            out_returns_tan_cum.append(out_returns_tan_cum[i-1]+((out_returns_tan[i]/100)))      

out_data= [3.64,2.13,1.27,1.84,-1.58,1.11,3.93,3.65,-1.56,-3.33,1.34,3.54,2.12,-4.49,-0.6,-6.15,-2.96,-0.76,4.78,2.04,-8.27,-0.62,1.66,-9.09,-17.15,-7.83,1.74,-8.12,-10.09,8.97,10.2,5.21,0.44,7.73,3.34,4.09,-2.59,5.56,2.76,-3.36,3.4,6.32,2.01,-7.88,-5.55,6.94,-4.76,9.55,3.89,0.61,6.83,2,3.5,0.46,2.9,-1.27,-1.75,-2.36,-5.98,-7.59,11.35,-0.28,0.74,5.05,4.42,3.11,-0.85,-6.18,3.89,0.79,2.56,2.74,-1.75,0.79,1.19,5.57,1.29,4.03,1.56,2.8,-1.2,5.65,-2.71,3.77,4.18,3.12,2.81,-3.32,4.65,0.43,-0.19,2.06,2.61,-2.04,4.24,-1.97,2.52,2.55,-0.06,-3.11,6.13,-1.12,0.59,1.36,-1.53,1.54,-6.04,-3.07,7.75,0.56,-2.16,-5.76,-0.05,6.98,0.92,1.79,-0.02,3.97,0.51,0.27]

out_market_cum = []
for i in range(119):
    if i == 0:
        out_market_cum.append((out_data[i])*(1+(out_data[i+1]/100)))
    else:
        if out_market_cum[i-1] > 0:
            out_market_cum.append(out_market_cum[i-1]*(1+(out_data[i]/100)))
        else:
            out_market_cum.append(out_market_cum[i-1]+((out_data[i]/100)))
        
#plot cumulative returns
pylab.figure(figsize=(12,6))
pylab.plot(out_returns_gmvp_cum, label='GMVP',lw=2)
pylab.plot(out_returns_tan_cum,label = 'Tangent',lw=2)
pylab.plot(out_market_cum,label = 'Market',lw=2)
pylab.xlabel('Month')
pylab.ylabel('Cumulative Monthly Returns')
pylab.grid(True)
pylab.title('Cumulative Adjusted weights returns')
pylab.legend(loc = 'upper left')
pylab.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Cumulative OOS Returns.jpeg', bbox_inches = 'tight')


##############Efficient Frontier
##################
##################
def statistics(GMVP_weights):
    weights = np.array(GMVP_weights)
    pret = np.sum(rets240.mean()* GMVP_weights) * 12
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets240.cov() *12, weights)))
    return np.array([pret, pvol, (pret-0.822)/pvol]) # assumes rf=2.4%, need to fix, 3rd entry returns sharpe ratio
   
import scipy.optimize as sco

def min_sharpe(GMVP_weights):
    return -statistics(GMVP_weights)[2]
    
cons=({"type":"eq","fun":lambda	x:np.sum(x)-1}) #all parameters sum to 1
bnds=tuple((-1,	1)for x	in range(10)) #weights are betwwen -100% and 100%

optimised = sco.minimize(min_sharpe, 10 * [1./10], method = 'SLSQP', bounds = bnds, constraints = cons)
optimised
optimised['x'].round(3)

statistics(optimised['x']).round(3)

#minimise variance
def min_var(GMVP_weights): 
    return statistics(GMVP_weights)[1]**2
    
optvar = sco.minimize(min_var, 10 * [1./10],method = 'SLSQP', bounds=bnds, constraints = cons)
optvar
optvar['x'].round(3)
statistics(optvar['x']).round(3)

#efficient frontier
cons = ({'type': 'eq', 'fun':lambda x: statistics(x)[0] - tret},{'type':'eq','fun':lambda x: np.sum(x)-1})
bnds = tuple((-1,1) for x in GMVP_weights)

def min_port(GMVP_weights):
    return statistics(GMVP_weights)[1]
    
trets = np.linspace(0,35,60)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun':lambda x: statistics(x)[0] - tret},{'type':'eq','fun':lambda x: np.sum(x)-1})
    res = sco.minimize(min_port, 10 * [1./10], method = 'SLSQP', bounds = bnds, constraints = cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

#capital market line
import scipy.interpolate as sci
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)

def f(x):
    return sci.splev(x, tck, der=0)
    
def df(x):
    return sci.splev(x, tck, der=1)
    
def equations(p, rf=0.822):
    eq1 = rf - p[0]
    eq2 = rf +p[1]*p[2]-f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3
    
opt = abs(sco.fsolve(equations, [0.822, 50, 15]))
opt # first entry should be risk free rate

np.round(equations(opt), 9)#want equations = 0

cx = np.linspace(0.0, 30)
cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0]-f(opt[2])},{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
res = sco.minimize(min_port, 10*[1./10], method = 'SLSQP', bounds = bnds, constraints=cons)
res['x'].round(3)

#http://matplotlib.org/users/text_intro.html
plt.figure(figsize=(12,6))
plt.scatter(tvols, trets, c= 'maroon', marker='x') #efficient frontier
plt.scatter(rets240['Food '].std()*np.sqrt(12), rbar[0]*12/100, c= ((rbar[0]*12/100)-0.822)/rets240['Food '].std()*np.sqrt(12), marker ='o')
plt.annotate('Food',xy=(rets240['Food '].std()*np.sqrt(12),(rbar[0]*12/100)-0.822))
plt.scatter(rets240['Hshld'].std()*np.sqrt(12), rbar[1]*12/100, c= ((rbar[1]*12/100)-0.822)/rets240['Hshld'].std()*np.sqrt(12), marker ='o')
plt.annotate('Hshld',xy=(rets240['Hshld'].std()*np.sqrt(12),(rbar[1]*12/100)-0.822))
plt.scatter(rets240['Drugs'].std()*np.sqrt(12), rbar[2]*12/100, c= ((rbar[2]*12/100)-0.822)/rets240['Drugs'].std()*np.sqrt(12), marker ='o')
plt.annotate('Drugs',xy=(rets240['Drugs'].std()*np.sqrt(12),(rbar[2]*12/100)-0.822))
plt.scatter(rets240['Chems'].std()*np.sqrt(12), rbar[3]*12/100, c= ((rbar[3]*12/100)-0.822)/rets240['Chems'].std()*np.sqrt(12), marker ='o')
plt.annotate('Chems',xy=(rets240['Chems'].std()*np.sqrt(12),(rbar[3]*12/100)-0.822))
plt.scatter(rets240['Txtls'].std()*np.sqrt(12), rbar[4]*12/100, c= ((rbar[4]*12/100)-0.822)/rets240['Txtls'].std()*np.sqrt(12), marker ='o')
plt.annotate('Txtls',xy=(rets240['Txtls'].std()*np.sqrt(12),(rbar[4]*12/100)-0.822))
plt.scatter(rets240['Aero '].std()*np.sqrt(12), (rbar[5]*12/100), c= ((rbar[5]*12/100)-0.822)/rets240['Aero '].std()*np.sqrt(12), marker ='o')
plt.annotate('Aero',xy=(rets240['Aero '].std()*np.sqrt(12),(rbar[5]*12/100)-0.822))
plt.scatter(rets240['Util '].std()*np.sqrt(12), (rbar[6]*12/100), c= ((rbar[6]*12/100)-0.822)/rets240['Util '].std()*np.sqrt(12), marker ='o')
plt.annotate('Util',xy=(rets240['Util '].std()*np.sqrt(12),(rbar[6]*12/100)-0.822))
plt.scatter(rets240['Softw'].std()*np.sqrt(12), rbar[7]*12/100, c= ((rbar[7]*12/100)-0.822)/rets240['Softw'].std()*np.sqrt(12), marker ='o')
plt.annotate('Softw',xy=(rets240['Softw'].std()*np.sqrt(12),(rbar[7]*12/100)-0.822))
plt.scatter(rets240['Fin  '].std()*np.sqrt(12), rbar[8]*12/100, c= ((rbar[8]*12/100)-0.822)/rets240['Fin  '].std()*np.sqrt(12), marker ='o')
plt.annotate('Fin',xy=(rets240['Fin  '].std()*np.sqrt(12),(rbar[8]*12/100)-0.822))
plt.scatter(rets240['RlEst'].std()*np.sqrt(12), rbar[9]*12/100, c= ((rbar[9]*12/100)-0.822)/rets240['RlEst'].std()*np.sqrt(12), marker ='o')
plt.annotate('RlEst',xy=(rets240['RlEst'].std()*np.sqrt(12),(rbar[9]*12/100)-0.822))
plt.plot(statistics(optimised['x'])[1],statistics(optimised['x'])[0], 'g*', markersize=15.0,label='Tangent Portfolio')#highest sharpe ratio - green star Er= 23.05%, SD = 13.86%
plt.plot(statistics(optvar['x'])[1],statistics(optvar['x'])[0],'r*',markersize=15.0,label='GMVP')#GMVP - red star Er = 12%  SD=9.96%
plt.grid(True)									
plt.xlabel('Expected Volatility')									
plt.ylabel('Expected Return')
plt.title('Out of Sample Efficient Frontier')									
plt.plot(cx, opt[0] + opt[1]*cx, lw=1.5,label='Capital Allocation Line')
plt.legend(loc='upper left', numpoints=1)
plt.axis([0, 40, 0, 40])
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/OOS Frontier.jpeg', bbox_inches = 'tight')

####################FF TEST
out_mv = pd.read_csv('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Data.csv')
index_out = rets240.index
df_data_gmvp = {'out_returns_gmvp': pd.Series(out_returns_gmvp,index = index_out), 'RF': out_mv['RF'], 'MktlessRF': out_mv['Mkt-RF'], 'SMB': out_mv['SMB'], 'HML': out_mv['HML']}
df_model_data_gmvp = DataFrame(df_data_gmvp, index = index_out)

df_data_tan = {'out_returns_tan': pd.Series(out_returns_tan, index = index_out), 'RF': out_mv['RF'], 'MktlessRF': out_mv['Mkt-RF'], 'SMB': out_mv['SMB'], 'HML': out_mv['HML']}
df_model_data_tan = DataFrame(df_data_tan, index = index_out)

import statsmodels.formula.api as smf
mod_tan_CAPM = smf.ols(formula='out_returns_tan - RF ~ MktlessRF', data=df_model_data_tan)
out_returns_tan - out_mv.ix[:,3]
res_tan_CAPM = mod_tan_CAPM.fit()
res_tan_CAPM.summary()

mod_tan = smf.ols(formula='out_returns_tan - RF ~ MktlessRF + HML', data=df_model_data_tan)
res_tan = mod_tan.fit()
res_tan.summary()

####Summary statistics
tangent = pd.DataFrame(out_returns_tan)
tangent.describe()
GMVP = pd.DataFrame(out_returns_gmvp)
GMVP.describe()
0.026883*12
4.769006*np.sqrt(12)
0.837101*12
4.132328*np.sqrt(12)
1.142233*12
4.388144*np.sqrt(12)
0.620109*12
2.923830*np.sqrt(12)
import scipy.stats as stats
stats.kurtosis(GMVP,fisher=True)
stats.kurtosis(tangent)

stats.skew(GMVP)
stats.skew(tangent)

#Sharpe
(((np.mean(GMVP))-(0.822/12))/(np.std(GMVP)))*np.sqrt(12)#0.647015
(((np.mean(tangent))-(0.822/12))/(np.std(tangent)))*np.sqrt(12)#-0.030356

################
################Stochastic Dominance 
#FOSD
sorted_GMVP = np.sort(out_returns_gmvp)
yvals=np.arange(len(sorted_GMVP))/float(len(sorted_GMVP)-1)
plt.plot(sorted_GMVP,yvals)
plt.show()

sorted_tang = np.sort(out_returns_tan)
plt.plot(sorted_GMVP,yvals, lw=2.5, label = 'GMVP',c='b')
plt.plot(sorted_tang,yvals, lw=2.5, label='Tangent',c='g')
plt.title('GMVP vs Tangent')
plt.xlabel('Returns')
plt.ylabel('Cumulative Probability')
plt.legend(loc='upper left')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/GMVP vs Tangent.jpeg')


#Food ###########
food_out = rets240['Food ']
food_sort = np.sort(food_out)
plt.plot(sorted_GMVP,yvals, lw=2.5, label = 'GMVP',c='b')
plt.plot(sorted_tang,yvals, lw=2.5, label='Tangent',c='g')
plt.plot(food_sort,yvals, lw=2.5, label ='Food',c='r')
plt.title('Food Stochastic Dominance')
plt.xlabel('Returns')
plt.ylabel('Cumulative Probability')
plt.legend(loc='upper left')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Food.jpeg')
	
#Hshld	
Hshld_out = rets240['Hshld']
Hshld_sort = np.sort(Hshld_out)
plt.plot(Hshld_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)
#Drugs
drugs_out = rets240['Drugs']
drugs_sort = np.sort(drugs_out)
plt.plot(drugs_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)	
#Chems
chems_out = rets240['Chems']
chems_sort = np.sort(chems_out)
plt.plot(chems_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)	
#Txtls
txtls_out = rets240['Txtls']
txtls_sort = np.sort(txtls_out)
plt.plot(txtls_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)
#Aero
aero_out = rets240['Aero ']
aero_sort = np.sort(aero_out)
plt.plot(aero_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)
#Util
util_out = rets240['Util ']
util_sort = np.sort(util_out)
plt.plot(util_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5) 	
#Softw	
softw_out = rets240['Softw']
softw_sort = np.sort(softw_out)
plt.plot(softw_sort,yvals, lw=2.5)
plt.plot(sorted_tang,yvals, lw=2.5)
#Fin
fin_out = rets240['Fin  ']
fin_sort = np.sort(fin_out)
plt.plot(fin_sort,yvals, lw=2.5,label='Fin',c='m')
plt.plot(sorted_GMVP,yvals, lw=2.5, label = 'GMVP',c='b')
plt.plot(sorted_tang,yvals, lw=2.5, label='Tangent',c='g')
plt.title('Finance Stochastic Dominance')
plt.xlabel('Returns')
plt.ylabel('Cumulative Probability')
plt.legend(loc='upper left')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Finance Stochastic.jpeg')	
#RlEst
rlest_out = rets240['RlEst']
rlest_sort = np.sort(rlest_out)
plt.plot(rlest_sort,yvals, lw=2.5,label='RlEst',c='y')
plt.plot(sorted_GMVP,yvals, lw=2.5, label = 'GMVP',c='b')
plt.plot(sorted_tang,yvals, lw=2.5, label='Tangent',c='g')
plt.title('Real Estate Stochastic Dominance')
plt.xlabel('Returns')
plt.ylabel('Cumulative Probability')
plt.legend(loc='upper left')
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/Real Estate.jpeg')


#Graph
plt.figure(figsize=(12,6))
plt.plot(sorted_GMVP,yvals,lw=4,label='GMVP')
plt.plot(sorted_tang,yvals,lw=4,label='Tangent')
plt.plot(food_sort,yvals,label='Food')
plt.plot(Hshld_sort,yvals,label='Hshld')
plt.plot(drugs_sort,yvals,label='Drugs')
plt.plot(chems_sort,yvals,label='Chems')
plt.plot(txtls_sort,yvals,label='Txtls')
plt.plot(aero_sort,yvals,label='Aero')
plt.plot(util_sort,yvals,label='Util')
plt.plot(softw_sort,yvals,label='Softw')
plt.plot(fin_sort,yvals,label='Fin')
plt.plot(rlest_sort,yvals,label='RlEst')
plt.title('First Order Stochastic Dominance')
plt.xlabel('Returns')
plt.ylabel('Cumulative Probability')
plt.legend(loc='upper left')
plt.axis([-30, 30, 0, 1])
plt.savefig('C:\Users\User\Documents\Actuary\Fourth Year\Financial Economics I\Project/FOSD.jpeg', bbox_inches = 'tight')

#SUBPLOTS
plt.subplot(2,2,1)
plt.plot(sorted_GMVP,yvals,lw=4,label='GMVP')
plt.plot(food_sort,yvals,lw=2,label='Food')
plt.plot(Hshld_sort,yvals,lw=2,label='Hshld')
plt.plot(drugs_sort,yvals,lw=2,label='Drugs')
plt.plot(chems_sort,yvals,lw=2,label='Chems')
plt.plot(txtls_sort,yvals,lw=2,label='Txtls')
plt.legend(loc='lower right',bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)

plt.subplot(2,2,2)
plt.plot(sorted_tang,yvals,lw=2,label='Tangent')
plt.plot(food_sort,yvals,lw=2,label='Food')
plt.plot(Hshld_sort,yvals,lw=2,label='Hshld')
plt.plot(drugs_sort,yvals,lw=2,label='Drugs')
plt.plot(chems_sort,yvals,lw=2,label='Chems')
plt.plot(txtls_sort,yvals,lw=2,label='Txtls')


plt.subplot(2,2,3)
plt.plot(sorted_GMVP,yvals,lw=2,label='GMVP')
plt.plot(aero_sort,yvals,lw=2,label='Aero')
plt.plot(util_sort,yvals,lw=2,label='Util')
plt.plot(softw_sort,yvals,lw=2,label='Softw')
plt.plot(fin_sort,yvals,lw=2,label='Fin')
plt.plot(rlest_sort,yvals,lw=2,label='RlEst')


plt.subplot(2,2,4)
plt.plot(sorted_tang,yvals,lw=2,label='Tangent')
plt.plot(aero_sort,yvals,lw=2,label='Aero')
plt.plot(util_sort,yvals,lw=2,label='Util')
plt.plot(softw_sort,yvals,lw=2,label='Softw')
plt.plot(fin_sort,yvals,lw=2,label='Fin')
plt.plot(rlest_sort,yvals,lw=2,label='RlEst')
plt.legend(loc='lower right',bbox_to_anchor=[1, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)


#SOSD
from statsmodels.distributions import ECDF
ecdf = ECDF(out_returns_gmvp)
y = ecdf(sorted_GMVP)
plt.step(sorted_GMVP,y)
plt.show()

from scipy import integrate
def f(x):
    res = x
    return res

Y = ecdf(sorted_GMVP)

#plot(X,f(X))

def F(x):
    res = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(f,0,val)
        res[i]=y
    return res

ecdf_aero = ECDF(aero_out)
ecdf_aero(aero_sort)

ecdf_fin = ECDF(fin_out)
integral_fin = np.cumsum(ecdf_fin(fin_sort))
integral_gmvp = np.cumsum(ecdf(sorted_GMVP))

plt.plot(ecdf_aero(aero_sort),F(ecdf_aero(aero_sort)))
plt.plot(fin_sort,integral_fin)
plt.plot(sorted_GMVP,integral_gmvp)

#SOSD #https://github.com/jamlamberti/Py4FinOpt/blob/master/02_Stochastic_Dominance.ipynb
from collections import defaultdict
import numpy as np

def compute_pdf(time_series):
    d = sorted(time_series)
    di = defaultdict(int)
    inc = 1.0/len(d)
    
    for i in range(len(d)):
        di[d[i]] += 1
    
    val  = []
    prob = []
    
    for k in sorted(di.keys()):
        val.append(k)
        prob.append(inc*di[k])

    return val, prob

def compute_cdf(prob):
    cdf = [prob[0]]
    for i in range(1, len(prob)):
        cdf.append(prob[i] + cdf[i-1])
    return cdf
    
val, pdf = compute_pdf(out_returns_gmvp)
cdf = compute_cdf(pdf)
print pdf
print cdf

def expand_vector(events, x, y):
    index = 0
    d_mod = []
    for pnt in events:
        if index >= len(x):
            d_mod.append(y[-1])
        elif pnt < x[index]:
            if index == 0:
                d_mod.append(0.0)
            else:
                d_mod.append(y[index-1])
        elif pnt == x[index]:
            d_mod.append(y[index])
        else:
            index += 1
            if index >= len(x):
                d_mod.append(y[-1])
            elif x[index] == pnt:
                d_mod.append(y[index])
            else:
                d_mod.append(y[index-1])
    return d_mod
    
def check_fosd(d1, d2):
    val1, pdf1 = compute_pdf(d1)
    val2, pdf2 = compute_pdf(d2)
    cdf1, cdf2 = map(compute_cdf, [pdf1, pdf2])
    points = sorted(list(set(val1+val2)))
    d1_mod = map(lambda x: round(x, 5), expand_vector(points, val1, cdf1))
    d2_mod = map(lambda x: round(x, 5), expand_vector(points, val2, cdf2))
    d1_fosd_d2 = all(map(lambda x, y: x<=y, d1_mod, d2_mod))
    d2_fosd_d1 = all(map(lambda x, y: x>=y, d1_mod, d2_mod))
    return d1_fosd_d2, d2_fosd_d1
    
def check_sosd(d1, d2):
    val1, pdf1 = compute_pdf(d1)
    val2, pdf2 = compute_pdf(d2)
    cdf1, cdf2 = map(compute_cdf, [pdf1, pdf2])
    points = sorted(list(set(val1+val2)))
    d1_mod = map(lambda x: round(x, 5), expand_vector(points, val1, cdf1))
    d2_mod = map(lambda x: round(x, 5), expand_vector(points, val2, cdf2))
    d1_areas = np.cumsum([d1_mod[i]*(points[i+1]-points[i]) for i in range(len(points)-1)])
    d2_areas = np.cumsum([d2_mod[i]*(points[i+1]-points[i]) for i in range(len(points)-1)])
    d1_sosd_d2 = all(map(lambda x, y: x<=y, d1_areas, d2_areas))
    d2_sosd_d1 = all(map(lambda x, y: x>=y, d1_areas, d2_areas))
    return d1_sosd_d2, d2_sosd_d1

print check_fosd(out_returns_gmvp, out_returns_tan)
print check_sosd(out_returns_gmvp, out_returns_tan)#GMVP SOSD Tang
#Food
print check_sosd(out_returns_gmvp, food_out)#Food SOSD GMVP
print check_sosd(out_returns_tan, food_out)#Food SOSD Tang
#Hshld
print check_sosd(out_returns_gmvp, Hshld_out)
print check_sosd(out_returns_tan, Hshld_out)#Hshld SOSD Tang
#Drugs
print check_sosd(out_returns_gmvp, drugs_out)
print check_sosd(out_returns_tan, drugs_out)#Drugs SOSD Tang
#Chems
print check_sosd(out_returns_gmvp, chems_out)
print check_sosd(out_returns_tan, chems_out)
#Aero
print check_sosd(out_returns_gmvp, aero_out)
print check_sosd(out_returns_tan, aero_out)#Aero SOSD Tang
#Txtls
print check_sosd(out_returns_gmvp, txtls_out)
print check_sosd(out_returns_tan, txtls_out)
#Util
print check_sosd(out_returns_gmvp, util_out)
print check_sosd(out_returns_tan, util_out)#Util SOSD Tang
#Softw
print check_sosd(out_returns_gmvp, softw_out)
print check_sosd(out_returns_tan, softw_out)
#Fin
print check_sosd(out_returns_gmvp, fin_out)#GMVP SOSD Fin
print check_sosd(out_returns_tan, fin_out)#Fin SOSD Tang
#RlEst
print check_sosd(out_returns_gmvp, rlest_out)#GMVP SOSD RlEst
print check_sosd(out_returns_tan, rlest_out)