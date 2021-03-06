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


for type in (0,1):
    print(type)

df[df['tArea']==2]


list(df)

#%% Plot admit coefficient by sample
for type in range(1,5):
    for idx,samp in enumerate(['Close', 'VeryClose']):
        # Read data and import dict of variable labels:
        dta = os.path.join(estsdir, 'Target_1b_wSample'+samp+'RD_sAll_vltotinc_tc_gAll.dta')
        reader = pd.io.stata.StataReader(dta)
        with reader as f:
            labs = f.variable_labels()
        # Replace empty values in labs dict with original varname
        for key, value in labs.items():
            if not value:
                labs.update({key: key})
        # Create dataframe and apply value labels
        df = pd.read_stata(dta)
        df = df.rename(index=str, columns=labs)
        reader.close()
        # Plot admit coefficients over target Area, by Sample
        y = df['tArea']
        plt.errorbar(df['_b[Type_'+str(type)+'_x_admit]'], y, xerr=df['_se[Type_'+str(type)+'_x_admit]'], marker='o', ls='none', label=samp)
        plt.axvline(x=0, linewidth=1, color='grey')
        # plt.yticks(df.index.astype(float), Area_labels)
        plt.legend()
    plt.title(r'$admit \cdot $'+str(type)+'.$Type$')
    ax = plt.axes()
    ax.yaxis.grid()
    plt.show()
    plt.close()

# itr = pd.read_stata(os.path.join(estsdir,'OLS_Basic1b_All_ltotinc_tc_All.dta'), iterator = True)
# itr.variable_labels()
# df.reset_index(inplace=True)
# # Plot coefficients
# for degree in ['','2']:
#     # math and read in same graph:
#     for idx,coef in enumerate(['math', 'read']):
#         plt.errorbar(df['_b_'+coef+degree], df.index+idx/5, xerr=df['_se_'+coef+degree]*zscore, marker='o', ls='none', label=coef+degree, alpha=0.7)
#         plt.axvline(x=0, linewidth=1, color='grey')
#         plt.yticks(df.index, Area_labels)
#         plt.legend()
#         plt.tight_layout()
#     plt.show()






#%% 2a
# Import and label data:
df = pd.read_stata(os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta'))
itr = pd.read_stata(os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta'), iterator = True)
df = df.rename(index=str, columns = itr.variable_labels())

df = ReshapeUnstacked(df)

## Plots

# Plot subset of coefficients
# coefs = ['math', 'math2', 'read', 'read2', 'exp', 'exp2', '1.Type', '2.Type', '3.Type']
# df2 = df[(df.coef.isin(coefs)) ]
# plt.errorbar(x=df2.beta, y=df2.coef, xerr=df2.se*zscore, ls='none', marker='o')
# plt.axvline(x=0, linewidth=1, color='grey')
# plt.tight_layout()
# plt.show()

# Plot University fixed effects
UList = pd.read_excel(os.path.join(inputsdir, 'Lists.xls'), sheet_name='U List', header=None, names=['Ucode', 'Uname'])
UList['Uname'] = UList['Uname'].str.replace('UNIVERSIDAD', 'U.')
df2 = df[df['coef'].str.contains('tUniv')]
# df2['Ucode'] = df2.coef.str.extract(r'(\d+)', expand=False)
df2 = df2.join(df2.coef.str.extract(r'(\d+)', expand=True))
df2.rename(index=str, columns={0:'Ucode'}, inplace=True)
df2['Ucode'] = df2['Ucode'].astype(int)
df2 = df2.merge(UList, left_on='Ucode', right_on='Ucode', how='left')

fig = plt.figure(0, figsize=(10,6))
plt.errorbar(x=df2.beta, y=df2.Uname, xerr=df2.se*zscore, ls='none', marker='o')

# plt.figure(figsize=(10,5))
# plt.errorbar(x=df2.beta, y=df2.Uname, xerr=df2.se*zscore, ls='none', marker='o')
# plt.axvline(x=0, linewidth=1, color='grey')
# plt.tight_layout()
# ax = plt.axes()
# ax.yaxis.grid()
# plt.show()




#%% Test
# Import and label data:
file = os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta')
reader = pd.io.stata.StataReader(file)
with reader as f:
    labs = f.variable_labels()

df = pd.read_stata(dta)
df = df.rename(index=str, columns=labs)

df

# df = pd.read_stata(os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta'))
# labs = pd.io.stata.StataReader(os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta')).variable_labels()




# itr = pd.read_stata(os.path.join(estsdir,'OLS_Basic2a_All_ltotinc_tc_All.dta'), iterator = True)
# df = df.rename(index=str, columns = itr.variable_labels())








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


df = pd.read_stata(os.path.join(estsdir, 'OLS_Basic3b_All_ltotinc_tc_All.dta'))

df
