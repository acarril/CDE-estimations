#%% Preamble
import numpy as np;
import seaborn as sns;
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
import scipy.stats as st
import os
import statsmodels.formula.api as sm
import math

#%% Function
from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize

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

#%% Something
for coef,q in {'math':'Pquant', 'read':'Pqual'}.items():
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
#    plt.errorbar(df['_b_'+coef],df['ratioQuant'], xerr=df['_se_'+coef]*1.96, marker=None, ls='none', label=coef, alpha=.5)
    plt.scatter(df['_b_'+coef],df[q], c=df['t_'+coef], cmap='viridis', norm=MidpointNormalize(midpoint=1.96))
    plt.axvline(x=0, linewidth=1, color='grey')
    plt.title(coef)
    plt.ylabel('Proportion of ' + q[1:] + ' courses')
    plt.colorbar(label='$t$-stat', ticks=[math.ceil(min(df['t_'+coef])), zscore,math.floor(max(df['t_'+coef]))])
    plt.tight_layout()
    plt.show()
