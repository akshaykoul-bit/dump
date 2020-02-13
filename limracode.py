# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:57:09 2019

@author: akshkoul
"""
import os
import pandas as pd
import numpy as np
import datetime
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LassoCV

os.getcwd()

filepath=r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra'

# Read data
limra_df=pd.read_csv(os.path.join(filepath,'pp_cap-Final.csv'))


# Re-coding StatusCode into a binary variable
limra_df['StatusCode']=limra_df['StatusCode'].replace([0,2,3,4],0)

#limhead=limra_df.tail(30000)

# Creating a local copy
limra_clean=limra_df.copy()

# Creating columns for attained ages for Primary and Secondary Owners
limra_clean['PrimDOB'].replace(np.nan,0,inplace=True)
limra_clean['SecDOB'].replace(np.nan,0,inplace=True)
limra_clean['TermDate'].replace(np.nan,0,inplace=True)

# Attained age columns
def attained_age(colname,newcolname):
    age_ls=[]
    start=time.time() 
    for index,i in enumerate(limra_clean[colname]):
        if i != 0:
            age_ls.append(limra_clean['ObsYr'][index]-datetime.datetime.strptime(str(i),'%m/%d/%Y').year)
        else:
            age_ls.append(np.nan)
    end=time.time() 
    print('Conversion Time: %f'%(float(end)-float(start)))
    limra_clean[newcolname]=age_ls

attained_age('PrimDOB','PrimAge_attained')
attained_age('SecDOB','SecAge_attained')

# Issued Age columns
def issued_age(colname,newcolname):
    age_ls=[]
    start=time.time() 
    for index,i in enumerate(limra_clean[colname]):
        if i != 0:
            age_ls.append(datetime.datetime.strptime(limra_clean['IssueDate'][index],'%m/%d/%Y').year-datetime.datetime.strptime(str(i),'%m/%d/%Y').year)
        else:
            age_ls.append(np.nan)
    end=time.time() 
    print('Conversion Time: %f'%(float(end)-float(start)))
    limra_clean[newcolname]=age_ls

issued_age('PrimDOB','PrimAge_issued')
issued_age('SecDOB','SecAge_issued')

# Policy Term Duration
limra_clean['IssueDate'].isna().sum()
dur_ls=[]
start=time.time() 
for index,i in enumerate(limra_clean['TermDate']):
    if i != 0:
        dur_ls.append(datetime.datetime.strptime(str(i),'%m/%d/%Y').year-datetime.datetime.strptime(limra_clean['IssueDate'][index],'%m/%d/%Y').year)
    else:
        dur_ls.append(np.nan)
end=time.time() 
print('Conversion Time: %f'%(float(end)-float(start)))
limra_clean['PolicyTermDuration']=dur_ls


#p=pd.to_datetime(limra_clean['IssueDate'][0])-pd.to_datetime(limra_clean['PrimDOB'][0])
#year(p)
#limra_clean['LoanBOY'].value_counts()
limclean_head=limra_clean.head(1000)


# Re-coding Risk Class as per the new mapping provided by LIMRA
riskmap=pd.read_excel('Downloads/Risk Class Mapping - company to standard industry categories.xlsx')
riskmap=riskmap.iloc[3:,1:3].reset_index().drop('index',axis=1)
limra_clean['PrimRiskClass'].replace(riskmap.iloc[:,0].values,riskmap.iloc[:,1].values,inplace=True)
limra_clean['SecRiskClass'].replace(riskmap.iloc[:,0].values,riskmap.iloc[:,1].values,inplace=True)
riskmap=pd.read_excel('Downloads/Risk Class Mapping - company to standard industry categories.xlsx')
riskmap=riskmap.iloc[3:,1:3].reset_index().drop('index',axis=1)
limra_clean['PrimRiskClass'].replace(riskmap.iloc[:,0].astype(str).values,riskmap.iloc[:,1].values,inplace=True)
limra_clean['SecRiskClass'].replace(riskmap.iloc[:,0].astype(str).values,riskmap.iloc[:,1].values,inplace=True)

# Exporting to drive
destination_folder=r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra'
#limra_clean.to_csv(os.path.join(destination_folder,'limra_clean.csv'))
limra_clean['PrimRiskClass'].value_counts()
limra_clean['SecRiskClass'].value_counts()

# Filtering out negative ages/unidentified genders
limra_clean=limra_clean[limra_clean['PrimAge_attained']>=0]
limra_clean=limra_clean[limra_clean['PrimAge_issued']>=0]
limra_clean=limra_clean[limra_clean['PrimGender'] != 'U']
#limra_clean1=limra_clean.drop_duplicates()
#limra_clean1=limra_clean1[limra_clean1['DistChannel'] !="Unknown"]
#limra_clean1=limra_clean1[limra_clean1['PrimRiskClass'] !="Unknown"]
#limra_clean1=limra_clean1[limra_clean1['SecRiskClass'] !="Unknown"]
#limra_clean1=limra_clean1[limra_clean1['ProdType'] !="other_premprod"]

#limra_clean1['ExtMatOpt'].dropna(inplace=True)


# Creating columns for preliminary analysis
limra_clean['ProdType'].replace([1,2,3,4,5,6,7],['ul_sg','ul_accum','iul_sg','iul_accum','vul_sg','vul_accum','other_premprod'],inplace=True)
limra_clean['DistChannel'].replace([0,1,2,3,4,5,6],['Unknown','Career Agent/Multiline Agent','Independent Agent','Wirehouse','Bank','Financial Planner','Other'],inplace=True)
bins=[-1,19,29,39,49,59,100]
labels=['Under 20','20-29','30-39','40-49','50-59','60 and over']
limra_clean['PrimAgeGrp_attained']=pd.cut(limra_clean['PrimAge_attained'],bins=bins,labels=labels)
limra_clean['SecAgeGrp_attained']=pd.cut(limra_clean['SecAge_attained'],bins=bins,labels=labels)
limra_clean['PrimAgeGrp_issued']=pd.cut(limra_clean['PrimAge_issued'],bins=bins,labels=labels)
limra_clean['SecAgeGrp_issued']=pd.cut(limra_clean['SecAge_issued'],bins=bins,labels=labels)
bins_pol=[-1,0,1,2,3,4,5,10,20,40]
labels_pol=['0','1','2','3','4','5',"6-10","11-20","21+"]
limra_clean['PolicyYearGrp']=pd.cut(limra_clean['Duration'],bins=bins_pol)

statemap=pd.read_csv(os.path.join(filepath,'states.csv'))
statemap=statemap.iloc[:51,1:3]
limra_clean['IssueRegion']=limra_clean['IssueState'].replace(statemap.iloc[:,0].values,statemap.iloc[:,1].values)
limra_clean['Smoker_Y_N']=limra_clean['PrimRiskClass'].replace(['Unknown', 'Substandard', 'Preferred Non Smoker', 'Standard Non Smoker', 'Standard Smoker', 'Preferred Smoker'],
           ['Unknown','Non-Smoker','Non-Smoker','Non-Smoker','Smoker','Smoker'])

limra_clean['ProductType']=limra_clean['ProdType'].replace(['ul_sg','ul_accum','iul_sg','iul_accum','vul_sg','vul_accum','other_premprod'],
           ['Universal Life','Universal Life','Indexed Universal Life','Indexed Universal Life',
            'Variable Universal Life','Variable Universal Life', 'Other Premium Product'])



# New cleaned dataset
limra_clean_new=limra_clean.drop(['PrimDOB','SecDOB','IssueDate','TermDate','SecAge_attained',
                               'SecGender','SecAge_issued','SecAgeGrp_attained','SecAgeGrp_issued',
                               'SecRiskClass','SecGuarInd','PolicyNo','FundingPattern','Zip','PolicyTermDuration'],axis=1)

limra_clean_new.to_csv(os.path.join(destination_folder,'limra_clean_new.csv'),index=False)
lhead=limra_clean_new.head(5000)
limra_clean_new=pd.read_csv(os.path.join(filepath,'limra_clean_new.csv'))
limra_clean_mod=limra_clean_new[limra_clean_new['CurrentPrem'].notna()==True]   
lmod=limra_clean_mod.head(5000)
limra_clean_mod.drop(['PrimAgeGrp_attained',
       'PrimAgeGrp_issued', 'PolicyYearGrp', 'Smoker_Y_N',
       'ProductType'],axis=1,inplace=True)

# Exporting dataset to csv file
limra_clean_mod.to_csv(os.path.join(destination_folder,'limra_clean_mod.csv'),index=False)

#limra_clean_mod.IssueRegion.value_counts()

# Creating a class to generate EDA plots
class lapR:
    
    xcol1=" "
    lcol2=" "
    prodtype_ls_full=[]
    productype=" "
    kind= " "
    ast=pd.DataFrame()
    a=pd.DataFrame()
    a1=pd.DataFrame()
    b=pd.DataFrame()
    c=pd.DataFrame()
    
    def grouped_data(self):
        self.a=self.ast[self.ast['ProductType']==(self.productype)]
        self.b=limra_clean_new.groupby([self.xcol1,self.lcol2]).count()['StatusCode']
        self.a1=self.a.groupby([self.xcol1,self.lcol2]).count()['StatusCode']
        self.c=self.a1/self.b*100

    def lapse_rates_grouped(self):
        fig, ax = plt.subplots(figsize=(15,7))
        self.c.unstack().plot(kind=self.kind,ax=ax)
        plt.ylabel("Lapse Rates")
        ax.set_xticklabels(labels_pol)
        plt.title("Lapse Rates for {1} over {0}\n {2}".format(self.xcol1,self.lcol2,self.productype))
        plt.savefig(os.path.join(filepath,'{0}_vs_{1}_grouped_{2}_{3}.png'.format(self.xcol1,self.lcol2,self.productype,self.kind)))
        plt.close()
    
    def complete_data(self):
        self.b=limra_clean_new.groupby([self.xcol1,self.lcol2]).count()['StatusCode']
        self.a1=self.ast.groupby([self.xcol1,self.lcol2]).count()['StatusCode']
        self.c=self.a1/self.b*100   

    def lapse_rates_complete(self):
        fig, ax = plt.subplots(figsize=(15,7))
        self.c.unstack().plot(kind=self.kind,ax=ax)
        plt.ylabel("Lapse Rates")
        ax.set_xticklabels(labels_pol)
        plt.title("Lapse Rates for {1} over {0}".format(self.xcol1,self.lcol2))
        plt.savefig(os.path.join(filepath,'{0}_vs_{1}_{2}.png'.format(self.xcol1,self.lcol2,self.kind)))
        plt.close()
 

lp=lapR()
lp.ast=limra_clean_new[limra_clean_new['StatusCode']==1] 

for colname in ["IssueRegion","PrimAgeGrp_issued","PrimAgeGrp_attained","PrimGender","PrimRiskClass","DistChannel","CoCode","Smoker_Y_N","ObsYr"]:    
    lp.xcol1= "PolicyYearGrp"
    lp.lcol2= colname
    lp.prodtype_ls_full= ['Universal Life','Indexed Universal Life','Variable Universal Life','Other Premium Product']
    for i in range(len(lp.prodtype_ls_full)):
        lp.productype=lp.prodtype_ls_full[i]    
        lp.grouped_data()
        for j in ["line","bar"]:
            lp.kind=j
            lp.lapse_rates_grouped()

lp.xcol1= "PolicyYearGrp"
lp.lcol2= "Smoker_Y_N"
lp.prodtype_ls_full= ['Universal Life','Indexed Universal Life','Variable Universal Life','Other Premium Product']
for i in range(len(lp.prodtype_ls_full)):
    lp.productype=lp.prodtype_ls_full[i]    
    lp.grouped_data()
    lp.kind="bar"
    lp.lapse_rates_grouped()

for colname in ["IssueRegion","PrimAgeGrp_issued","PrimAgeGrp_attained","PrimGender","PrimRiskClass","DistChannel","CoCode","Smoker_Y_N","ObsYr"]:    
    lp.xcol1= "PolicyYearGrp"
    lp.lcol2= "ProductType"
    lp.complete_data()
    for j in ["line","bar"]:
        lp.kind=j
        lp.lapse_rates_complete()

    lp.prodtype_ls=lp.prodtype_ls_full[0]    
    lp.grouped_data()
    for j in ["line","bar"]:
        lp.kind=j
    lp.lapse_rates_grouped()

    lp.productype=lp.prodtype_ls_full[1]    
    lp.grouped_data()
    lp.kind="bar"
    lp.lapse_rates_grouped()

#l=labels_pol
#print(l)
#limra_clean_new.FundingPattern.value_counts()
#
#limclean_head3=limra_clean_new.head(5000)
#
#l=limra_clean_new.PolicyYearGrp.unique().tolist()
#l.sort()
#for col in ["IssueRegion","PrimAgeGrp_issued","PrimAgeGrp_attained","PrimGender","PrimRiskClass","DistChannel","CoCode","Smoker_Y_N","ObsYr",'ProductType']:
#    fig, ax = plt.subplots(figsize=(15,7))
#    limra_clean_new.groupby(['PolicyYearGrp',col]).sum()['CumPrem'].unstack().plot(kind='line',ax=ax)
#    ax.set_xticklabels(labels_pol)
#    plt.title("Cummulative Premium Paid over Poilcy Years according to {col}".format(col=col))
#    plt.savefig(os.path.join(filepath,'CumPrem_{col}_line.png'.format(col=col)))
#    plt.close()
#
#    col="ProductType"
#    fig, ax = plt.subplots(figsize=(15,7))
#    limra_clean_new.groupby(['PolicyYearGrp',col]).sum()['CurrentPrem'].unstack().plot(kind='bar',ax=ax)
#    ax.set_xticklabels(labels_pol)
#    plt.title("Current Year Premium according to {col}".format(col=col))
#    plt.savefig(os.path.join(filepath,'CurrPrem_{col}_bar.png'.format(col=col)))
#    plt.close()
#
#n={}
#for i in limra_df.columns:
#    n[i]=limra_df[i].notna().sum()
#
#n.update(limra_df.notna().sum(axis = 0))
#
#
#lim_clean4=limra_clean_new[limra_clean_new['CurrentPrem'].notna()==True]
#limra_clean1['TermDate']



######## Model Building ########################################################

# Read data
limra_clean_new=pd.read_csv(os.path.join(filepath,'limra_clean_new.csv'))
lhead=limra_clean_new.head(5000)
#categorical variables
cat_var=['ObsYr','CoCode','ProdType','PrimGender','IssueState','DistChannel','PrimAgeGrp_attained','PrimAgeGrp_issued', 
         'PrimRiskClass','StatusCode','SecGuar','ExtMatOpt','LTCRider','CVEnRider',"IssueRegion","Smoker_Y_N"]
limcat=limra_clean_new[cat_var]
limcat_head=limcat.head(100)

# Target data set
limcat_tgt=limcat[limcat['StatusCode']==1].reset_index().drop(['index'],axis=1)
limtgt_head=limcat_tgt.head(1000)
#limcat_tgt.values
for col in limcat_tgt.columns:
    limcat_tgt[col].value_counts().plot.pie(subplots=True,autopct='%1.2f%%')
    plt.title('Count percentage of policies lapsed\nfor "{col}"'.format(col=col))
    plt.savefig(r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra\lapsecountperc_{col}.png'.format(col=col))
    plt.clf()

limra_clean_new["PrimAgeGrp_attained"].value_counts().plot.pie(subplots=True,autopct='%1.2f%%')
plt.title('Count percentage of policies lapsed\nfor "{col}"'.format(col=col))
plt.savefig(r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra\lapsecountperc_{col}.png'.format(col=col))
#plt.clf()   
# save figures
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

#for col in limcat_tgt.columns:
#    ax_lim=limcat_tgt[col].value_counts().plot.pie()
#    # create a list to collect the plt.patches data
#    totals = []
#    
#    # find the values and append to list
#    for j in ax_lim.patches:
#        totals.append(j.get_height())
#    
#    # set individual bar lables using above list
#    total = sum(totals)
#    
#    # set individual bar lables using above list
#    for j in ax_lim.patches:
#        # get_x pulls left or right; get_height pushes up or down
#        ax_lim.text(j.get_x()-.03, j.get_height()+.5, 
#                str(round((j.get_height()/total)*100,2))+'%',
#                    color='dimgrey')
#    plt.title('Percentage Distribution of "{col}" for policies terminated due to lapse'.format(col=col))
#    plt.savefig(r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra\target_{col}.png'.format(col=col))
#    plt.clf()


# PC/PP Ratios by Product Type
prodgrp=limra_clean_new.groupby(['ProductType']).agg({'CurrentPrem':{'total_currprem':'sum'},'AnnPlannedPrem':{'plannedprem':'sum'}})
prodgrp['ratio']=prodgrp.iloc[:,0]/prodgrp.iloc[:,1]
fig,ax=plt.subplots()
p1=plt.bar(prodgrp.index,prodgrp['ratio'],color='orange')
plt.title('PC/PP Ratio by Product Type')
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(round(height*100,3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(p1)
plt.tight_layout()
plt.show()

# PC/PP Ratios over Years
obsyrgrp=limra_clean_new.groupby(['ObsYr']).agg({'CurrentPrem':{'total_currprem':'sum'},'AnnPlannedPrem':{'plannedprem':'sum'}})
obsyrgrp['ratio']=obsyrgrp.iloc[:,0]/obsyrgrp.iloc[:,1]
fig,ax=plt.subplots()
y1=plt.bar(obsyrgrp.index,obsyrgrp['ratio'],color='green')
plt.title('PC/PP Ratio by Observered Year')
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(round(height*100,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(y1)
plt.tight_layout()
plt.show()



# Trend for PC/PP over Policy Year
pyrgrp=limra_clean_new.groupby(['Duration']).agg({'CurrentPrem':'sum','AnnPlannedPrem':'sum','CumPrem':'sum'})
pyrgrp['PC/PP ratio']=pyrgrp.iloc[:,0]/pyrgrp.iloc[:,1]
cp=[]
for i in range(len(pyrgrp['CurrentPrem'])):
    if i==0:
        cp.append(0)
    else:
        cp.append(pyrgrp['CurrentPrem'].iloc[i]/pyrgrp['CurrentPrem'].iloc[i-1])
pyrgrp['PC/PPr ratio']=cp

fig, ax = plt.subplots()
barwidth=0.45
r1=pyrgrp.index.tolist()
r2=[x+barwidth for x in r1]
b1=ax.bar(r1,pyrgrp['PC/PP ratio'],width=barwidth,color='brown')
b2=ax.bar(r2,pyrgrp['PC/PPr ratio'],width=barwidth,color='blue')
b3=pyrgrp['CurrentPrem'].plot(secondary_y=True,color='green')
b4=pyrgrp['AnnPlannedPrem'].plot(secondary_y=True,ax=ax,color='orange')
ax.legend((b1[0], b2[0]), ('Current Year Premium / Annual Planned Premium (Ratio)', 'Current Year Premium / Previous Year Premium (Ratio)'))
ax.set_xlabel("Duration (Policy Years)")
ax.set_title("Trends for Collected Premium v/s Planned Premium by policy year (duration)")
plt.xlim(1,25)
plt.xticks(np.arange(30))
plt.legend(bbox_to_anchor=(1,1), loc="upper left",fontsize='medium')
plt.show()

# Percentage Distribution for Statuscode=1 by Age
#limra_df['PrimRiskClass'].value_counts()
#limcat_tgt['PrimAge']= pd.to_datetime(limcat_tgt['IssueDate'])-pd.to_datetime(limcat_tgt['PrimDOB'])
#limcat_tgt['PrimAge'] = limcat_tgt['PrimAge'].astype('<m8[Y]')
#limtgt_head=limcat_tgt.head(25)
#ag=limcat_tgt[limcat_tgt['PrimAge']<=0]
#bins=[-10,0,10,25,40,55,70,100]
#labels=['Insured Before/On Birth','1-10','11-25','26-40','41-55','56-70','71+']
#limcat_tgt['PrimeAgeGrp']=pd.cut(limcat_tgt['PrimAge'],bins=bins,labels=labels)
axage=limra_clean_new['PrimAgeGrp_attained'].value_counts().plot(kind='bar')

## create a list to collect the plt.patches data
totals = []    

for j in axage.patches:
        totals.append(j.get_height())

## set individual bar lables using above list
total = sum(totals)
    
## set individual bar lables using above list
for j in axage.patches:
    ### get_x pulls left or right; get_height pushes up or down
    axage.text(j.get_x()-.03, j.get_height()+.5, 
                str(round((j.get_height()/total)*100,2))+'%',
                    color='dimgrey')
plt.title('Percentage Distribution of "PrimAgeGrp_attained" for policies terminated due to lapse')
plt.setp(axage.get_xticklabels(), rotation=0, horizontalalignment='right',fontsize='medium' )
plt.show()

# PC/PP Ratio for different Distribution Channels
distgrp=limra_clean1.groupby(['DistChannel']).agg({'CurrentPrem':{'total_currprem':'sum'},'AnnPlannedPrem':{'plannedprem':'sum'}})
distgrp['ratio']=distgrp.iloc[:,0]/distgrp.iloc[:,1]
fig,ax=plt.subplots()
y1=plt.bar(distgrp.index,distgrp['ratio'],color='maroon')
plt.title('PC/PP Ratio for different Distribution Channels')
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(round(height*100,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(y1)
plt.tight_layout()
plt.show()

ast=limra_clean_new[limra_clean_new['StatusCode']==1]
b=limra_clean_new.groupby('IssueRegion').count()['StatusCode']
a1=ast.groupby('IssueRegion').count()['StatusCode']
c=a1/b*100
fig,ax=plt.subplots()
c1=c.plot(kind='bar',ax=ax)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(round(height*100,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(c1)
#plt.tight_layout()
plt.show()
def grouped_data(col):
    b=limra_clean_new.groupby(col).count()['StatusCode']
    a1=ast.groupby(col).count()['StatusCode']
    c=a1/b*100
    fig, ax = plt.subplots(figsize=(15,7))
    c=c.plot(kind='bar',ax=ax)
    plt.ylabel("Lapse Rates")
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}%'.format(round(height*100,2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(c)
#    ax.set_xticklabels(labels_pol)
    plt.title("Lapse Rates for {col}".format(col=col))
    plt.tight_layout()
    plt.show()
#    plt.savefig(os.path.join(filepath,'{0}_vs_{1}_grouped_{2}_{3}.png'.format(xcol1,lcol2,productype,kind)))
#    plt.close()
grouped_data('IssueState')

# XGBoost model 
import time
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score
import seaborn as sns


limra_clean_mod=pd.read_csv(os.path.join(filepath,'limra_clean_mod.csv'))
limra_clean_mod.ProdForm.value_counts()
len(limra_clean_mod[limra_clean_mod.TargetPrem < 0])
col_obj=limra_clean_mod.select_dtypes("O").columns.tolist()
#
#from sklearn.impute import SimpleImputer
#imp_mean = SimpleImputer( strategy='most_frequent')
#imp_mean.fit(limra_clean_mod[col_obj])
#imputed_train_df = imp_mean.transform(train)
#
#import sys
#from impyute.imputation.cs import fast_knn
#sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
#
### start the KNN training
##imputed_training=fast_knn(limra_clean_mod[['LoanBOY']].astype('float64').values, k=30)
##
##
##
##limra_clean_mod.isna().sum()
##sns.pairplot(limra_clean_mod)
##sns.plt.show()
##
##
##column_mod=limra_clean_mod.select_dtypes("O").columns.tolist()
##limra_clean_mod.LoanBOY.value_counts()
##
##lmod=limra_clean_mod.drop(["GuarFamtBOY","SecGuar","ExtMatOpt"],axis=1)
##X=lmod.drop('StatusCode',axis=1)
##y=lmod['StatusCode']
##X['LoanBOY'].isna().sum()

# Label Encoding
l_en=preprocessing.LabelEncoder()
encod_dict={}
def encoding(col):
    limra_clean_mod[col] = l_en.fit_transform(limra_clean_mod[col].astype(str))
    encod_dict[col]= dict(zip(l_en.classes_, l_en.transform(l_en.classes_)))

for col in column_mod:
    encoding(col)

# Imputation
from sklearn.impute import SimpleImputer
imr=SimpleImputer(missing_values=np.nan, strategy='mean')
col_imp=['AVBOY',
 'FamtBOY',
 'CumPrem',
 'AnnPlannedPrem',
 'CreditedRateBOY',
 'GuarCreditedRate',
 'TargetPrem',
 'PrimAge_attained',
 'PrimAge_issued']



mean=X[col_imp].mean(axis=0)
for i in col_imp:
    X[i] = X[i].fillna(X[i].mean(axis=0))

X.isna().sum()


train=pd.read_csv(os.path.join(filepath,'train.csv'))
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


train.dtypes

# Over sampling
from imblearn.over_sampling import SMOTENC
sm = SMOTENC(categorical_features=list(range(8,19)),random_state=7)

train=pd.read_csv(os.path.join(filepath,'train.csv'))
test=pd.read_csv(os.path.join(filepath,'test.csv'))
len(test.ProdForm.unique())
train.columns
#train=train.iloc[:,1:]
#test=test.iloc[:,1:]
X_train=train.drop('StatusCode',axis=1)
y_train=train.StatusCode
X_test=test.drop('StatusCode',axis=1)
y_test=test.StatusCode
#columns = X_train.columns
sm_data_X,sm_data_y=sm.fit_resample(X_train, y_train)
sm_data_X = pd.DataFrame(data=sm_data_X,columns=columns )
sm_data_y= pd.DataFrame(data=sm_data_y,columns=['StatusCode'])
#sm_data_y.StatusCode.value_counts()

len(sm_data_X.ProdType.unique())
len(sm_data_X.ProdForm.unique())
smote_train= sm_data_X.copy()
smote_train["StatusCode"]=sm_data_y["StatusCode"]
#smote_train.columns

# we can Check the numbers of our data
print("length of oversampled data is ",len(sm_data_X))
print("Number of in-force in oversampled data",len(sm_data_y[sm_data_y['StatusCode']==0]))
print("Number of lapsed",len(sm_data_y[sm_data_y['StatusCode']==1]))
print("Proportion of in-force data in oversampled data is ",len(sm_data_y[sm_data_y['StatusCode']==0])/len(sm_data_X))
print("Proportion of lapsed data in oversampled data is ",len(sm_data_y[sm_data_y['StatusCode']==1])/len(sm_data_X))

# Export oversampled data
smote_train.to_csv(r'C:\Users\akshkoul\Documents\Fall19\Capstonelimra\smote_train3.csv',index=False)


# fit model
model = XGBClassifier()
start1= time.time()
model.fit(sm_data_X, sm_data_y)
end1= time.time()
print('Execution time for fit is: %f'%(float(end1)-float(start1)))

# make predictions for test data
start2=time.time()
y_pred = model.predict(X_test)
end2=time.time()
print('Execution time for prediction is: %f'%(float(end2)-float(start2)))


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

print("Recall: %.2f" %recall_score(y_test,y_pred))

# Defining a function for ROC Plot
def roc_plot(false_pr, true_pr):
    plt.plot(false_pr, true_pr, color='green', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

false_pr, true_pr, thresholds = roc_curve(y_test, probs)
roc_plot(false_pr, true_pr)

model.feature_importances_.plot(kind='bar')

# plot feature importance
from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(model)
pyplot.show()

# RandomForest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=7)
start2=time.time()
clf.fit(sm_data_X, sm_data_y)  
end2=time.time()
print('Execution time for fit is: %f'%(float(end2)-float(start2)))

start2=time.time()
y_pred2 = clf.predict(X_test)
end2=time.time()
print('Execution time for prediction is: %f'%(float(end2)-float(start2)))


# evaluate predictions
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy: %.2f%%" % (accuracy2 * 100.0))

probs2 = clf.predict_proba(X_test)
probs2 = probs2[:, 1]
auc2 = roc_auc_score(y_test, probs2)
print('AUC: %.2f' % auc2)

print("Recall: %.2f" %recall_score(y_test,y_pred2))

print(plt.plot(clf.feature_importances_))


# plot
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()