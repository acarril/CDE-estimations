#%% Preamble
import numpy as np;
import seaborn as sns;
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st
import os
import statsmodels.formula.api as sm
estsdir = 'estimations'
inputsdir = 'inputs'
Area_labels = ["Business and Administration",
               "Agriculture",
               "Architecture and Art",
               "Natural Sciences",
               "Social Sciences",
               "Law",
               "Education",
               "Humanities",
               "Health",
               "Technology"]

#%% Import data
df = pd.read_stata('estimations/OLS_Basic2b_All_ltotinc_tc_All.dta')
FL = pd.read_stata(os.path.join(inputsdir,'mallas.dta'))
df = df.merge(FL, left_on='tFLcode_app', right_on='FLcode_app', how='left')
len(df[(df['Cquant']>df['duration'])])
df['ratioQuant'] = df['Cquant']/df['duration']
df['ratioQuant'].describe()

#%% Process
df['ratioQuant'] = df['Cquant']/df['duration']
df = df[['tFLcode_app', '_b_math', '_se_math', 'Cquant', 'ratioQuant']]
df['ci_math'] = df['_se_math']*1.96;
df['lb_math'] = df['_b_math'] - (df['ci_math'])
df['ub_math'] = df['_b_math'] + (df['ci_math'])
df['signf_math'] = (df['lb_math'] > 0) | (df['ub_math'] < 0)

#%% Plot coefficients
for coef in ['math']:
    # Remove outlier
    df = df.loc[df['_b_'+coef] > -2]
    df = df.sort_values(by=['ratioQuant'])
    df = df.reset_index(drop=True)
    df = df.loc[df['_b_'+coef] > df['_b_'+coef].quantile(q=0.025)]
#    plt.errorbar(df['_b_'+coef],df['ratioQuant'], xerr=df['_se_'+coef]*1.96, marker=None, ls='none', label=coef, alpha=.5)
    plt.scatter(df['_b_'+coef],df['ratioQuant'], c=df['signf_math'])
    plt.axvline(x=0, linewidth=1, color='grey')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% OLS
ols = sm.ols(formula="ratioQuant ~ _b_math", data=df).fit()
ols.summary()
