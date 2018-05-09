#%% Preamble
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st
import seaborn as sns
import statsmodels.formula.api as sm
from matplotlib import cbook, colors
from matplotlib.colors import Normalize

#%% Function
from numpy import ma


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#%% Define important variables
zscore = st.norm.ppf(.975) # Note that 1.96 is the zscore inside of which is 95% of the data (ignoring both tails), but st.norm.ppf() gives the zscore which has 95% of the data below it (ignoring only the upper tail).
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

################################################################################

#%% 1b: math, read
fig = plt.figure(figsize=(10, 4))
for i, (coef, q) in enumerate({'math':'Pquant', 'read':'Pqual'}.items()):
    # Import estimations:
    df = pd.read_stata(os.path.join(estsdir, 'OLS_Basic2b_All_ltotinc_tc_All.dta'))

    # Import and merge FL codes information:
    FL = pd.read_stata(os.path.join(inputsdir, 'mallas.dta'))
    df = df.merge(FL, left_on='tFLcode_app', right_on='FLcode_app', how='left')

    # Add confidence intervals, t-stats and significance dummy:
    df['ci_'+coef] = df['_se_'+coef]*zscore;
    df['lb_'+coef] = df['_b_'+coef] - df['ci_'+coef]
    df['ub_'+coef] = df['_b_'+coef] + df['ci_'+coef]
    df['t_'+coef] = abs(df['_b_'+coef]) / df['_se_'+coef]
    df['signf_'+coef] = (df['lb_'+coef] > 0) | (df['ub_'+coef] < 0)

    # Remove outlier
    df = df.loc[df['_b_'+coef] > -2]
    df = df.sort_values(by=[q])
    df = df.reset_index(drop=True)
    df = df.loc[df['_b_'+coef] > df['_b_'+coef].quantile(q=0.025)]

    # Plot:
    ax = fig.add_subplot(1,2,i+1)
    scat = ax.scatter(df['_b_'+coef],df[q], c=df['t_'+coef], cmap='viridis', norm=MidpointNormalize(midpoint=1.96))
    fig.colorbar(scat, label='$t$-stat', ticks=[math.ceil(min(df['t_'+coef])), zscore,math.floor(max(df['t_'+coef]))])
    ax.axvline(x=0, linewidth=1, color='grey')
    plt.title(coef)
    plt.ylabel('Proportion of ' + q[1:] + ' courses')

plt.tight_layout()
fig.subplots_adjust(wspace=.5)
plt.show()



#%% 1c: math, read
fig = plt.figure(figsize=(12, 4))
for i, (coef, q) in enumerate({'math':'Pquant', 'read':'Pqual'}.items()):
    # Import estimations:
    df = pd.read_stata(os.path.join(estsdir, 'OLS_Basic2c_All_ltotinc_tc_All.dta'))

    # Import and merge FL codes information:
    FL = pd.read_stata(os.path.join(inputsdir, 'mallas.dta'))
    df = df.merge(FL, left_on='tFLcode_app', right_on='FLcode_app', how='left')

    # Add confidence intervals, t-stats and significance dummy:
    df['ci_'+coef] = df['_se_'+coef]*zscore;
    df['lb_'+coef] = df['_b_'+coef] - df['ci_'+coef]
    df['ub_'+coef] = df['_b_'+coef] + df['ci_'+coef]
    df['t_'+coef] = abs(df['_b_'+coef]) / df['_se_'+coef]
    df['signf_'+coef] = (df['lb_'+coef] > 0) | (df['ub_'+coef] < 0)

    # Remove outlier
    df = df.loc[(df['_b_'+coef] > df['_b_'+coef].quantile(q=0.025)) & (df['_b_'+coef] < df['_b_'+coef].quantile(q=0.975))]
    df = df.sort_values(by=[q])
    df = df.reset_index(drop=True)

    # Plot:
    ax = fig.add_subplot(1,2,i+1)
    scat = ax.scatter(df['_b_'+coef],df[q], c=df['t_'+coef], cmap='viridis', norm=MidpointNormalize(midpoint=1.96))
    fig.colorbar(scat, label='$t$-stat', ticks=[math.ceil(min(df['t_'+coef])), zscore,math.floor(max(df['t_'+coef]))])
    ax.axvline(x=0, linewidth=1, color='grey')
    plt.title(coef)
    plt.ylabel('Proportion of ' + q[1:] + ' courses')

plt.tight_layout()
fig.subplots_adjust(wspace=.5)
plt.show()


df[(df['tsel_q'] == 1)]
