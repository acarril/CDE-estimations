import shutil
import os

homedir = 'C:/Users/alvaro/'
ests_dir = os.path.join(homedir,'Dropbox/CDE_light/3. Data/4. SII Outputs/UI/tables/')
repo_dir = os.path.join(homedir,'CDE-estimations')

#%% OLS
files = [f for f in os.listdir(ests_dir+'2. OLS/') if f.endswith(".dta")]
for f in files:
    shutil.copy2(os.path.join(ests_dir,'2. OLS/',f), os.path.join(repo_dir,'estimations'))
