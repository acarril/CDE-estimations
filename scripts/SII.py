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

areas = {1:'Business',2:'Agriculture',3:'Architecture and Art',4:'Natural Sciences',5:'Social Sciences',6:'Law',7:'Eduction',8:'Humanities',10:'Health',11:'Technology'}
################################################################################

# OLS
#%% Prepare datasets
# Estimates:
df = pd.read_stata(os.path.join(estsdir,'ols_basic0b_all_ltotinc_tc_all.dta'))
df = df.filter(like='tflcode_b')
df = df.T
df['FL'] = df.index.str.extract('(\d+)', expand=False).astype(int)
df.rename(index=str,columns={0:'beta'}, inplace=True)
# Crosswalk options:
co = pd.read_stata(os.path.join('inputs','crosswalk_options_April2018.dta'))
co = co.loc[co['proceso']==2003]
co = co[['FLcode_app','Area','Pquant','Pqual','Cquant','Cqual']].groupby('FLcode_app').mean()
df = df.merge(right=co, how='right', left_on='FL', right_index=True)
df.dropna(subset = ['beta'], inplace=True)

#%% FL FE by (subset of) areas
fig, ax = plt.subplots()
for area in [3,7,10,11]:
    sns.kdeplot(df.loc[df['Area']==area].beta, ax=ax, shade=True, label=areas.get(area));
ax.set_xlabel('Career-University FE')
plt.savefig('figs/FL-FE-byarea.png')

#%% All areas by quant terciles
for courses in ['Cquant','Pquant']:
    df['{0}_group'.format(courses)] = pd.qcut(df[courses], 3, labels=['Low','Medium','High'])
    fig, ax = plt.subplots()
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='Low'].beta, ax=ax, shade=True, label='Low');
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='Medium'].beta, ax=ax, shade=True, label='Medium');
    sns.kdeplot(df.loc[df['{0}_group'.format(courses)]=='High'].beta, ax=ax, shade=True, label='High');
    ax.set_xlabel('Career-University FE')
    ax.set_ylabel(courses)
    ax.set_title('All areas')
    plt.savefig('figs/FL-FE-by{0}3.png'.format(courses))

#%% All areas by Pquant terciles
df['Pquant_group'] = pd.qcut(df.Pquant, 3, labels=['Low','Medium','High'])
fig, ax = plt.subplots()
sns.kdeplot(df.loc[df['Pquant_group']=='Low'].beta, ax=ax, shade=True, label='Low');
sns.kdeplot(df.loc[df['Pquant_group']=='Medium'].beta, ax=ax, shade=True, label='Medium');
sns.kdeplot(df.loc[df['Pquant_group']=='High'].beta, ax=ax, shade=True, label='High');
ax.set_xlabel('Career-University FE')
ax.set_title('All areas')
plt.savefig('figs/FL-FE-byquant3.png')

#%% All areas by Pquant median
df['Pquant_group'] = pd.qcut(df.Pquant, 2, labels=['Low','High'])
fig, ax = plt.subplots()
sns.kdeplot(df.loc[df['Pquant_group']=='Low'].beta, ax=ax, shade=True, label='Low');
sns.kdeplot(df.loc[df['Pquant_group']=='High'].beta, ax=ax, shade=True, label='High');
ax.set_xlabel('Career-University FE')
ax.set_title('All areas')
plt.savefig('figs/FL-FE-byquant2.png')



for key,value in {k: areas[k] for k in (3,7,10,11)}.items():
    df2 = df.loc[df['Area']==key]
    try:
        df2['Pquant_group'] = pd.qcut(df2.Pquant, 2, labels=['Low','High'])
    except:
        pass
    else:
        fig, ax = plt.subplots()
        sns.kdeplot(df2.loc[(df['Area']==key) & (df['Pquant_group']=='Low')].beta, ax=ax, shade=True, label='Low');
    #    sns.kdeplot(df2.loc[(df['Area']==key) & (df['Pquant_group']=='Medium')].beta, ax=ax, shade=True, label='Medium');
        sns.kdeplot(df2.loc[(df['Area']==key) & (df['Pquant_group']=='High')].beta, ax=ax, shade=True, label='High');
        ax.set_xlabel('Career-University FE')
        ax.set_title('{0}'.format(value))
        plt.savefig('figs/FL-FE-byquant-{0}.png'.format(value))

#%%
for key,value in {k: areas[k] for k in (1,2,10,11)}.items():
    df2 = df.loc[df['Area']==key]
    df2['Pquant_group'] = pd.qcut(df2.Pquant, 2, labels=['Low','High'])
    fig, ax = plt.subplots()
    sns.kdeplot(df2.loc[(df['Area']==key) & (df['Pquant_group']=='Low')].beta, ax=ax, shade=True, label='Low');
    sns.kdeplot(df2.loc[(df['Area']==key) & (df['Pquant_group']=='High')].beta, ax=ax, shade=True, label='High');
    ax.set_xlabel('Career-University FE')
    ax.set_title('{0}'.format(value))
    plt.savefig('figs/FL-FE-byquant-{0}.png'.format(value))