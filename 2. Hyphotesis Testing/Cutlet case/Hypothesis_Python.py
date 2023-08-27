import pandas as pd # yang ini library untuk Datasheet # takda ni python tak leh baca
import scipy  # yang ini library untuk math dan statistic punya function
from scipy import stats #Dalam library scipy ni function stat untuk statistic
import statsmodels.api as sm # ni function utk semua model stat 

############ 2 sample T Test ##################

# Load the data
prom = pd.read_csv ("C:Users\\user\\Desktop\\data Sc with R & Pyhton\\Cutlets.csv")
prom

prom.columns = "InterestRateWaiver", "StandardPromotion"

# Normality Test
stats.shapiro(prom.InterestRateWaiver) # Shapiro Test
print(stats.shapiro(prom.StandardPromotion))
help(stats.shapiro)

# Variance test
scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.levene)
# p-value = 0.287 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.ttest_ind)


############# One - Way Anova ################
from statsmodels.formula.api import ols

cof=pd.read_excel("C:/Datasets_BA/360DigiTMG/DS_India/360DigiTMG DS India Module wise PPTs/Module 05 Hypothesis Testing/Data/ContractRenewal_Data(unstacked).xlsx")
cof
cof.columns="SupplierA","SupplierB","SupplierC"

# Normality Test
print(stats.shapiro(cof.SupplierA)) # Shapiro Test
print(stats.shapiro(cof.SupplierB))
print(stats.shapiro(cof.SupplierC))

# Variance test
scipy.stats.levene(cof.SupplierA, cof.SupplierB)
scipy.stats.levene(cof.SupplierB, cof.SupplierC)
scipy.stats.levene(cof.SupplierC, cof.SupplierA)

# One - Way Anova
mod = ols('SupplierA ~ SupplierB + SupplierC',data = cof).fit()

aov_table = sm.stats.anova_lm(mod, type=2)
help(sm.stats.anova_lm)

aov_table


######### 2-proportion test ###########
import numpy as np

two_prop_test = pd.read_csv("C:\\Users\\user\\Desktop\\data Sc with R & Pyhton\\JohnyTalkers.csv")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Person.value_counts()
tab1
tab2 = two_prop_test.Drinks.value_counts()
tab2

count = np.array([58,152])
nobs = np.array([480,740])

stats, pval = proportions_ztest(count, nobs, alternative='two-sided') 
print(pval) # Pvalue- 0.000  
stats, pval = proportions_ztest(count, nobs, alternative='larger')
print(pval)  # Pvalue 0.999  


################ Chi-Square Test ################

Bahaman=pd.read_excel("C:/Datasets_BA/360DigiTMG/DS_India/360DigiTMG DS India Module wise PPTs/Module 05 Hypothesis Testing/Data/Bahaman.xlsx")
Bahaman

count=pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count
Chisquares_results=scipy.stats.chi2_contingency(count)

Chi_square=[['', 'Test Statistic', 'p-value'],['Sample Data', Chisquares_results[0], Chisquares_results[1]]]
