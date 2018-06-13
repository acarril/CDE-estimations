#%% Preamble

# Modules:
import math
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st
import seaborn as sns
import statsmodels.formula.api as sm
from IPython.display import HTML
from matplotlib import cbook, colors
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

# Class: Normalize cmap
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Function: reshape unstacked dataset of results
def ReshapeUnstacked(df):
    # Transpose and reshape:
    df = df.T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'varname', '0': 'value'}, inplace=True)
    df['index'] = df['varname'].str.startswith('_se')
    df['index'] = df['index'].replace({True: 'se', False: 'beta'})
    df['varname'] = df['varname'].replace({'e(N)': 'e[N]'})
    df['coef'] = df['varname'].str.extract(r'((?<=\[).*(?=\]))', expand=True)
    df = df.pivot(index='coef', columns='index', values='value')
    df.reset_index(inplace=True)
    # Remove N of obs. from rows:
    N = df.at[df.loc[df['coef']=='N'].index[0], 'beta']
    df.drop(df[df.coef=='N'].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Important variables:
zscore = st.norm.ppf(.975) # Note that 1.96 is the zscore inside of which is 95% of the data (ignoring both tails), but st.norm.ppf() gives the zscore which has 95% of the data below it (ignoring only the upper tail).
estsdir = 'estimations/SII'
SIIests = os.path.join('estimations','SII')
areas = {1:'Business',2:'Agriculture',3:'Architecture and Art',4:'Natural Sciences',5:'Social Sciences',6:'Law',7:'Eduction',8:'Humanities',10:'Health',11:'Technology'}

################################################################################

# OLS
#%% Prepare dataset with estimated betas and se, merged with crosswalk options
# Estimated betas:
df = pd.read_stata(os.path.join(SIIests,'ols_basic0b_all_ltotinc_tc_all.dta'))
df = df.filter(like='tflcode_b')
df = df.T
df['FL'] = df.index.str.extract('(\d+)', expand=False).astype(int)
df.rename(index=str,columns={0:'beta'}, inplace=True)
# Estimated standard errors:
df_se = pd.read_stata(os.path.join(SIIests,'ols_basic0b_all_ltotinc_tc_all.dta'))
df_se = df_se.filter(like='tflcode_se')
df_se = df_se.T
df_se['FL'] = df_se.index.str.extract('(\d+)', expand=False).astype(int)
df_se.rename(index=str,columns={0:'se'}, inplace=True)
df = df.merge(right=df_se, on='FL')
# Crosswalk options:
co = pd.read_stata(os.path.join('inputs','crosswalk_options_April2018.dta'))
co = co.loc[co['proceso']==2003]
co = co[['FLcode_app','Area','Pquant','Pqual','Cquant','Cqual']].groupby('FLcode_app').mean()
df = df.merge(right=co, how='right', left_on='FL', right_index=True)
df.dropna(subset = ['beta'], inplace=True)

#%% Plot FL-FE by (subset of) areas
fig, ax = plt.subplots()
for area in [3,7,10,11]:
    sns.kdeplot(df.loc[df['Area']==area].beta, ax=ax, shade=True, label=areas.get(area));
ax.set_xlabel('Career-University FE')
plt.savefig('figs/FL-FE-byArea.png')

#%% Plot FL-FE of all areas by Cquant,Pquant terciles
# Notes: Cqual and Pqual don't seem to yield good results.
for courses in ['Cquant','Pquant']:
    df['{0}_group'.format(courses)] = pd.qcut(df[courses], 3, labels=['Low','Medium','High'])
    fig, ax = plt.subplots()
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='Low'].beta, ax=ax, shade=True, label='Low');
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='Medium'].beta, ax=ax, shade=True, label='Medium');
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='High'].beta, ax=ax, shade=True, label='High');
    ax.set_xlabel('Career-University FE')
    ax.legend(title=courses+' terciles:')
    plt.savefig(f'figs/FL-FE-by{courses}_q3.png')

#%% Plot FL-FE of all areas by Cquant,Pquant median
for courses in ['Cquant','Pquant']:
    df['{0}_group'.format(courses)] = pd.qcut(df[courses], 2, labels=['Low','High'])
    fig, ax = plt.subplots()
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='Low'].beta, ax=ax, shade=True, label='Low');
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='High'].beta, ax=ax, shade=True, label='High');
    ax.set_xlabel('Career-University FE')
    ax.legend(title=courses+' groups:')
    plt.savefig(f'figs/FL-FE-by{courses}_q2.png')

#%% Plot FL-FE of subset of areas by Cquant,Pquant median
for key,value in {k: areas[k] for k in (1,2,10,11)}.items():
    for courses in ['Cquant','Pquant']:
        df2 = df.loc[df['Area']==key]
        df2['{0}_group'.format(courses)] = pd.qcut(df2[courses], 2, labels=['Low','High'])
        fig, ax = plt.subplots()
        sns.kdeplot(df2.loc[(df['Area']==key) & (df['{0}_group'.format(courses)]=='Low')].beta, ax=ax, shade=True, label='Low');
        sns.kdeplot(df2.loc[(df['Area']==key) & (df['{0}_group'.format(courses)]=='High')].beta, ax=ax, shade=True, label='High');
        ax.set_xlabel('Career-University FE')
        ax.set_title('{0}'.format(value))
        ax.legend(title=courses+' groups:')
        plt.savefig(f'figs/FL-FE-by{courses}_q2-Area{key}.png')

#%%

ols = pd.read_stata(os.path.join('estimations','OLS_Basic1a_All_ltotinc_tc_All.dta'))
varlabs = pd.io.stata.StataReader(os.path.join('estimations','OLS_Basic1a_All_ltotinc_tc_All.dta')).variable_labels()
ols.rename(columns=varlabs, inplace=True)
ols = ols.filter(like='tArea')
ols = ols.filter(like='_b')
ols = ols.T
ols['area'] = ols.index.str.extract('(\d+)')
ols['area'] = ols['area'].astype(int)
ols = ols.reset_index(drop=True)
ols.rename(columns={0:'OLSbeta'}, inplace=True)
ols.area

from itertools import product
areas
df = pd.DataFrame(list(product(areas,areas)), columns=['tArea','snArea'])
df = df.merge(right=ols, left_on='tArea', right_on='area', copy=False)
df.rename(columns={'OLSbeta':'b_tArea'}, inplace=True)
df = df.merge(right=ols, left_on='snArea', right_on='area', copy=False)
df.rename(columns={'OLSbeta':'b_snArea'}, inplace=True)

df['b_OLS'] = df['b_tArea'] - df['b_snArea']
df.drop(columns=['area_x','area_y','b_tArea','b_snArea'], inplace=True)




rd = pd.read_stata(os.path.join('estimations','Target_0a_wSampleVeryCloseRD_sAll_vltotinc_tc_gAll.dta'), columns=['tArea','snArea','_b_admit'])
df = df.merge(right=rd)

df.replace(0, np.nan, inplace=True)
df.rename(columns={'_b_admit':'b_RD'}, inplace=True)

#df.to_csv(os.path.join('estimations','ols-rd.csv'))

df['b_OLS2'] = df['b_OLS'] + (df['b_RD'] - df['b_OLS'])/3

sel = pd.read_stata(os.path.join('inputs','AreaSelectivity.dta'))
df = df.merge(right=sel, on='tArea', copy=False)
df = df.merge(right=sel, left_on='snArea', right_on='tArea', copy=False)
df['selec'] = df['AveSelectivity_x'] + df['AveSelectivity_y']
df.drop(columns=['AveSelectivity_x', 'AveSelectivity_y', 'tArea_y'], inplace=True)
df

df['selec']

2+2

#%%
g = sns.regplot(x='b_OLS2',y='b_RD', data=df, label='Estimates', line_kws={'label':'Linear fit'})
g.text(.22,-1.1,'UI data (2000-2003)')
g.set_xlabel('OLS')
g.set_ylabel('RD')
g.set_title('RD by OLS prediction')
g.plot([1,-1],[1,-1], '--', label='45° line')
g.legend()
g.figure.savefig('figs/OLS-RD-prediction.pdf')

#%%

df['selq3'] = pd.qcut(df['selec'], 3, labels=['Low','Medium','High'])


for tercile in ['Low','Medium','High']:
    g = sns.regplot(x='b_OLS2',y='b_RD', data=df[df['selq3']==tercile], label='Estimates', line_kws={'label':'Linear fit'})
    g.text(.22,-1.1,'UI data (2000-2003)')
    g.set_xlabel('OLS')
    g.set_ylabel('RD')
    g.set_title(f'RD by OLS prediction - {tercile} selectivity')
    g.plot([1,-1],[1,-1], '--', label='45° line')
    g.legend()
    g.figure.savefig(f'figs/OLS-RD-prediction_q3{tercile}.png')
    g.figure.clf()

df


fig = sns.regplot(x='b_OLS2',y='b_RD', data=df.dropna())
# fig.text(df.b_OLS2[12]+0.2, df.b_RD[12], df.selec[12], horizontalalignment='left')
for line in range(1, 10):
    fig.annotate('hola', (df.dropna().b_OLS2[line], df.dropna().b_RD[line]))
