
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chisquare
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:


def ip_data_report(data = None , save_as_excel = 0):
    
    """Build an input data report from an excel file and store the report into a comma-separated values (csv) file if necessary

Parameters
----------
input_data : name of imported raw dataset
    
save_as_excel : 1 or 0, default 0
                1 - to save the report into a csv file,
                0 - if saving the file is not necessary               
"""

    variable = data.columns                                  #To get all the column names into a list
    dtype = data.dtypes                                      #To get all the data types of column into a list
    num_miss = data.isnull().sum()                           #To calculate number of missing data in each column and store it in a list
    miss_percent = 100 * data.isnull().sum() / len(data)     #To calculate missing percentage of the data in each column and store it in a list
    nunique_count = data.nunique()                           #To calculate the number of misisng values in each column
    num_rows = len(data)-num_miss                            #To calculate the number of non missing data in each column

    df = pd.DataFrame([variable, dtype, num_rows, num_miss, miss_percent, nunique_count])    #Building a data frame
    df = df.transpose()
    df.columns = ['variable', 'dtype', 'num_of_non-missing_rows', 'num_miss', 'miss_percent', 'nunique_count'] #Adding the header to each column of the data frame

    if save_as_excel == 1:
        df.to_csv('Input_Data_Report.csv')               #To save the dataframe into a CSV file
    else :   
        pd.options.display.max_rows = len(df.index)             #To display all the rows in the IDE
        pd.options.display.max_columns = len(df.columns)        #To display all the columns in the IDE
        print(df)
    return df


# In[4]:


def qc_type(data = None, df = None, cutoff = 50, save_as_excel = 0):
    
    """ Add the qc_type variable to the input data report and store it into a CSV file if neccesary.
    This variable QC type is used in create univariate & bivariate plots further down.
 
Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report 

cutoff : Cut off for number of levels of the variable. Above cutoff will be univariate 
    
save_as_excel : 1 or 0, default 0
                1 - to save the report into a csv file,
                0 - if saving the file is not necessary               
"""
   
    qc_type = []

 #-----------Loop to distinguish each column's type and adding it to the list----------
#------------------Loop Starts here---------------------------------------

    for x in range(0,len(df)):
        if df['dtype'][x] == 'object' :
            if df['nunique_count'][x] <= cutoff and df['nunique_count'][x] > 0:
                qc_type.append('freq dist')
                
            else:
                qc_type.append('no action')
            
        elif df['dtype'][x] == 'int64' or df['dtype'][x] == 'float64':
            if df['nunique_count'][x] > cutoff:
                qc_type.append('uni dist')
                
            elif df['nunique_count'][x] <= cutoff and df['nunique_count'][x] > 0:
                qc_type.append('freq dist')
                
            else:
                qc_type.append('no action')
        else:
            qc_type.append('no action')
            
#-----------------Loop Ends--------------------
            
    df['qc_type'] = qc_type                 # To add the list to the dataframe
    
    pd.options.display.max_rows = len(df.index)
    pd.options.display.max_columns = len(df.columns)
    
    print("Number of variable with 'no action' QC treatment :", df.loc[df['qc_type'] == 'no action', 'variable'].count())
    # df.loc[df['qc_type'] == 'no action', 'variable']
    
    print("\n")
    print("Number of variable with 'frequency' QC treatment :", df.loc[df['qc_type'] == 'freq dist', 'variable'].count())
    # df.loc[df['qc_type'] == 'freq dist', 'variable']
    
    print("\n")
    print("Number of variable with 'univariate' QC treatment :", df.loc[df['qc_type'] == 'uni dist', 'variable'].count())
    # df.loc[df['qc_type'] == 'uni dist', 'variable']
    
    if save_as_excel == 1:
        df.to_csv('Input_Data_Report_QC.csv')                #To save the updated dataframe into a new file 
    else:  
        pd.options.display.max_rows = len(df.index)             #To display all the rows in the IDE
        pd.options.display.max_columns = len(df.columns)        #To display all the columns in the IDE
        print(df)
    


# In[8]:


def uni_freq_plots(data = None, df = None):

    """ Create univariate and frequency plots and save it into seperate PDF's for Univariate and Frequency Distribution.
    And genereate the report of both the types of variables i.e. Univariate Distribution type and Frequency Distribution type into Univariate_distribution_Data_Report.csv and Frequency_distribution_Data_Report.csv
 
Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

"""

    counts = df['qc_type'].value_counts()
    p1 = PdfPages('Univariate_plots.pdf')                    #Create a new PDF 
    p2 = PdfPages('Frequency_distribution_plots.pdf')
    
    #--------------------------Reports of Univariate and Frequency plots------------------
    uni = df.loc[df['qc_type'] == 'uni dist']
    per = list(range(0,100,5))
    per +=([96,97,98,99,99.5,99.6,99.7,99.8,99.9])
    per = [x/100 for x in per ]
    K = data[uni['variable']].describe(percentiles = per)
    K.to_csv('Univariate_distribution_Data_Report.csv')
    
    K = pd.DataFrame()
    li = list()
    for x in range(0,len(df['qc_type'])):
        if df['qc_type'][x]=='freq dist':
            k=pd.DataFrame()
            k=pd.DataFrame(data[df['variable'][x]].fillna(value='Missing').value_counts())
            k.columns = ['Sample_size']
            k['Percentage_data'] = k['Sample_size']/k['Sample_size'].sum()
            li = [df["variable"][x]]*len(k)
            k['Variable'] = li
            K = pd.concat([K,k]) 
    K.index.names = ['Levels']
    K = K[['Variable','Sample_size','Percentage_data']]
    K.to_csv('Frequency_distribution_Data_Report.csv')
    
    
    
    #------------------Report ends-----------------------
    
    
    

#-------------------Loop to plot the graphs for Uni variate and frequency distribution into seperate files--------  
#------------------------------Loop Starts-------------------------------
    
    for x in range(0,len(df['qc_type'])):
        if df['qc_type'][x]=='uni dist':
            sns.set(style="white")
            a = sns.distplot(data[df['variable'][x]].dropna(), kde=False,color = '#009999')
            a.set(ylabel='Count')
            a.set_title(df['variable'][x])
            fig = plt.gcf()
            fig.set_size_inches(10, 8, forward=True)
            p1.savefig(fig)
            plt.clf()
        elif df['qc_type'][x]=='freq dist':
            k=data[df['variable'][x]].fillna(value='Missing').value_counts()
            sns.set(style="white")
            a = sns.barplot(x=k.index, y=100*k/k.sum(),color = '#009999')
            a.set(ylabel='Percentage')
            a.set_title(df['variable'][x])
            a.set_xticklabels(a.get_xticklabels(), fontsize=9, rotation =90)
            a.set_yticklabels(['{:.0f}%'.format(x) for x in a.get_yticks()])
            fig = plt.gcf()
            fig.set_size_inches(8, 12, forward=True)
            p2.savefig(fig)
            plt.clf()
#------------------------------Loop Ends------------------------
    p1.close()                                       # Close the file after after adding the plots into it
    p2.close()
    
#-------------------------To print Summary--------------
    print(counts['uni dist'], "Univariate plots are generated into a file named Univariate_plots.pdf ")
    print(counts['freq dist'],"Frequency distribution plots are generated into a file named Frequency_distribution_plots.pdf \nAnd no action is taken on remaining ",counts['no action'],"variables" )
    print('Reports for both the type of variables are genreated and saved into files named Univariate_distribution_Data_Report.csv and Frequency_distribution_Data_Report.csv')   


# In[7]:


def correlation(data=None):
    
    """Performs the Correlation on data to obtain the correlation matrix. The correlation matrix obtained is saved into a CSV file named Correlation Matrix.csv and plotted 
        
Parameters
----------
data : imported main dataset on which Correlation need to be performed

"""
    
    
    
    corr = data.corr()                          #Applying Correlation on the data
    
    fig = plt.figure() 
    plt.figure(figsize = (14,14))
    mask = np.zeros_like(corr,dtype = np.bool)          
    mask[np.triu_indices_from(mask)] = True
    sns.set(style="white")
    sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True,mask = mask, vmax=1, center=0, vmin=-1, linewidths=.01, cbar_kws={"shrink":.5}) #Plotting the correlation matrix
    #sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True)    
    plt.show()
    
    corr.to_csv('Correlation Matrix.csv') #Storing the correlation matrix into a CSV file
    
    
    


# In[ ]:


def Chi_Square(data=None,df = None, q=0.95):
    
    """Performs the Chi-Square Test on categorical data to find the similarity between all the unique levels in the data and obtain the variables list which passed the test. This list is stored into a CSV file named Chi Square Test - Null Hypothesis Passed.csv 
        
Parameters
----------
data : imported main dataset on which Chi-Square test need to be performed 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module
      
q  : Confidence level to perform Chi-Square Test.      
        
"""
    
    #----------------Loop to get all frequency distribution type into a list--------------
    fre=[]
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'freq dist':
            fre.append(df['variable'][i])
     #---------------------Loop ends-------------------------------       
    
    
    dummies = pd.DataFrame()                  #Creating a DataFrame
    
 #--------------------Loop to create dummies for all variables and storing it to the dataframe created above----------
    for x in fre:
        #if data[x].dtypes !='int64' and data[x].dtypes !='float64': 
        A = pd.get_dummies(data[x])
        for i in range(0,A.shape[1]) :
            A = A.rename(columns = {A.columns[i]:x + '_' + str(A.columns[i])})
        dummies=pd.concat([dummies,A],axis=1)
            
   #-------------------------------------Loop Ends--------------------------------------------------
            
   #----------------------------Performing Chi-Square test on each and every 2 unique levels or newly created variables--------
    li=[]

    cri = stats.chi2.ppf(q, df=1)
    for i in range(0,len(dummies.columns)):
        o = dummies[dummies.columns[i]].value_counts()
        for j in range(i+1,len(dummies.columns)):
            e=dummies[dummies.columns[j]].value_counts()
            chi_sqr_stat = (((o-e)**2)/e).sum()
            p_value = 1-stats.chi2.cdf(x=chi_sqr_stat,df=1)
            if chi_sqr_stat > cri :                       #Condition create a list with the variable only which passed the test
                t = {"Variable 1" : dummies.columns[i], "Variable 2" : dummies.columns[j]}#, "P Value" : p_value}
                li.append(t)
      #---------------------Loop Ends---------------------
    table = pd.DataFrame(li) # creating the dataframe with all the lists obtained
    table.to_csv('Chi Square Test - Null Hypothesis Passed.csv ')    #Storing the test results into a CSV file


# In[5]:


def bivariate_categorical_single_plot(data = None, df = None, TargetVar = None):
    
    """Plotting Bivariate graphs on categorical variables so as to analyse or find the best predictor. All the plots are saved into a PDF named Bivariate_Categorical_Single_Plot.pdf
This will create the chart with trend of the variable, average and sample size of each level in the variable.

Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

TargetVar : Name of the column of Target Variable in the data.

"""
    
    p = PdfPages('Bivariate_Categorical_Single_Plot.pdf')           #Opening a file 
    
    fre=[]
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'freq dist':
            fre.append(df['variable'][i])
    sns.set(style="white", palette="muted", color_codes=True)
    
    #----------------------- Loop to generate plots and save it into a PDF------------------- 
    for x in fre :
        if x!=TargetVar:
            Tv = data.groupby(x)
            N = pd.DataFrame([(Tv[TargetVar].sum()),(data[x].value_counts())])
            N = N.transpose()
            N['Rate']=100.0*N[TargetVar]/N[x]
            N['Average-Rate'] = [N[TargetVar].sum()*100/ N[x].sum()]*len(N)
            fig, ax1 = plt.subplots()
            fig.set_size_inches(8, 10, forward=True)
            ax2 = ax1.twinx()
            
            try:
                ax1.plot(N.index,N['Rate'],marker='o',color='#3399FF')
                ax1.plot(N.index,N['Average-Rate'],color='#202020',marker='*', linestyle='--', label = 'Average Rate')
            
            except:
                N = N.reset_index()
                N.columns = ['Levels',TargetVar,x,'Rate','Average-Rate']
                N.set_index('Levels')
                N['Levels'] = N.applymap(str)
                ax1.plot(N.index,N['Rate'],marker='o',color='#3399FF')
                ax1.plot(N.index,N['Average-Rate'],color='#202020',marker='*', linestyle='--', label = 'Average Rate')
                
            ax2=sns.barplot(x=N.index, y=N[x],alpha=0.60,color = '#00994C')
            for t in ax2.get_yticklabels():
                t.set_color('#00994C')
            for t in ax1.get_yticklabels():
                t.set_color('#3399FF')    
            ax1.legend(loc='best')
            ax1.set_ylabel('Rate',color='#3399FF')
            ax2.set_ylabel('Count',color='#00994C')
            ax1.set_title(x)
            ax1.set_xticklabels(N.index, fontsize=6.5, rotation =90)
            ax1.set_yticklabels(['{:.0f}%'.format(x) for x in ax1.get_yticks()],color = '#3399FF')
            p.savefig(fig)
            plt.clf()
        #-----------------------------Loop Ends-----------------------------
    
    p.close()            #Closing the file


# In[1]:


def bivariate_categorical_stack_plot(data = None, df = None, TargetVar = None):
    
    """Plotting Bivariate graphs on categorical variables so as to analyse or find the best predictor. All the plots are saved into a PDF named Bivariate_Categorical_Stack_plot.pdf
Creates stack plots per variable, one with trend of traget variable, average and other with sample size of levels in a variable.

Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

TargetVar : Name of the column of Target Variable in the data.

"""    
    
    fre=[]
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'freq dist':
            fre.append(df['variable'][i])
    p = PdfPages('Bivariate_Categorical_Stack_plot.pdf')                             #Opening a file
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set(rc={'figure.figsize':(16,8)})
    
    #----------------------- Loop to generate plots and save it into a PDF------------------- 
    
    
    for x in fre :
        if x!=TargetVar:
            Tv = data.groupby(x)
            N = pd.DataFrame([(Tv[TargetVar].sum()),(data[x].value_counts())])
            N = N.transpose()
            N['Percentage']=100.0*N[TargetVar]/N[x]
            N['Average'] = [N[TargetVar].sum()*100/ N[x].sum()]*len(N)
            fig = plt.figure()
            ax1 = plt.subplot(2, 1, 1)
            try:
                ax1.plot(N.index,N['Percentage'],marker='o',color='green')
                ax1.plot(N.index,N['Average'])
            except:
                N = N.reset_index()
                N.columns = ['Levels',TargetVar,x,'Percentage','Average']
                N.set_index('Levels')
                N['Levels'] = N.applymap(str)  
                ax1.plot(N.index,N['Percentage'],marker='o',color='green')
                ax1.plot(N.index,N['Average'])
            ax1.set_ylabel('Rate')
            ax1.set_xticklabels([],fontsize=8,rotation = 90)
            ax1.set_yticklabels(['{:.0f}%'.format(x) for x in ax1.get_yticks()])
        
            ax2 = plt.subplot(2, 1, 2)
            sns.barplot(x=N.index, y=N[x],alpha=0.75,color = '#00994C')
            plt.xticks(fontsize=8,rotation = 90)
            ax1.set_title(x)
            #plt.show()
            p.savefig(fig)
     #-----------------------------------Loop Ends-----------------------       
            
    p.close()            #Closing the file


# In[6]:


def bivariate_continuous(data = None, df = None, q = None, TargetVar = None):
    
    """Plotting Bivariate graphs on continuous variables so as to analyse or find the best predictor. All the plots are saved into a PDF named Bivariate_Continuous_Plot.pdf

Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

q  : Number of groups/deciles of data we need 

TargetVar : Name of the column of Target Variable in the data.

"""        

    uni=[]
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'uni dist':
            uni.append(df['variable'][i])

    sns.set(style="white", palette="muted", color_codes=True)
    
    p = PdfPages('Bivariate_Continuous_Plot.pdf')
    
    #---------------------------------Loop to geberate the plots-----------------------------
    
    for x in uni:
        if data[x].dtypes == 'int64' or data[x].dtypes == 'float64' :
            n = pd.DataFrame([data[x],data[TargetVar]])
            n = n.transpose()
            new = n.sort_values([x])
            new.columns = ['x','TargetVar']
            new['counter'] = range(0,len(new))
            new['decile'] = pd.qcut(new['counter'],q, labels=False)
            new1 = new.groupby(['decile']).TargetVar.mean()
            new2 = new.groupby(['decile']).counter.count()
            new3 = new.groupby(['decile']).x.min()
            new4 = new.groupby(['decile']).x.max()
            N = pd.concat([new1,new2,new3,new4],axis=1)
            N.columns = ['Mean','Count','1','2']
            N['Average-Rate'] = [N['Mean'].mean()]*len(N)
            fig, ax1 = plt.subplots()
            fig.set_size_inches(10, 8, forward=True)
            ax1.plot(N.index,N['Mean'],marker='o',color='#3399FF')
            ax1.plot(N.index,N['Average-Rate'],marker='*',linestyle='--',color='#202020')
            new5 = list()
            for y in range(0,len(N)):
                new5.append((N['1'][y],N['2'][y]))
            plt.xticks(N.index)
            ax1.set_xticklabels(new5,rotation = 90,fontsize=7)
            for t in ax1.get_yticklabels():
                t.set_color('#3399FF')
            ax1.set_title(x)
            ax1.legend(loc='best')
            ax1.set_ylabel('Rate',color='#3399FF')
            p.savefig(fig)
            plt.clf()
       #---------------------------Loops Ends---------------------------------------------------     
            
    p.close()        


# In[3]:


def Woe_IV_Categorical(data = None, df = None, TargetVar = None):
    
    """Find the WoE and IV values of each variable and each unique level of categorical data so as to find out the best predictor. These generated values are stored into a CSV named WoE_IV_Categorical.csv
Following is what the values of IV mean according to Siddiqi (2006):
    •Information Value Predictive Power 
    • < 0.02 useless for prediction 
    •0.02 to 0.1 Weak predictor 
    •0.1 to 0.3 Medium predictor 
    •0.3 to 0.5 Strong predictor 
    • > 0.5 Suspicious or too good to be true

Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

TargetVar : Name of the column of Target Variable in the data.

"""    
    
    fre = []
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'freq dist':
            fre.append(df['variable'][i])
    N2 = pd.DataFrame()
  #----------------------------------------Loop to perform Woe & Iv tests on every variable---------------------------  
    
    for x in fre:
        if x!=TargetVar:
            Tv = data.groupby(x)
            N = pd.DataFrame([(Tv[TargetVar].sum())])
            N = N.transpose()
            N.columns = ['No. of Goods']
            N1 = pd.DataFrame(data[x].value_counts())
            N = N.join(N1)
            N['No. of Bads'] = N[x]-N['No. of Goods']
            N.columns = ['Good','Total','Bad']
            N['Distribution of Good']=N['Good']/N['Good'].sum()
            N['Distribution of Bad']=N['Bad']/N['Bad'].sum()
            N['WoE'] = np.log(N['Distribution of Good']/N['Distribution of Bad'])
            N['IV'] = (N['Distribution of Good']-N['Distribution of Bad'])*N['WoE']
            N['Variable'] = [x]*len(N)
            N['Cummulative_IV'] = np.ma.masked_invalid(N['IV']).sum()
            N2 = pd.concat([N2,N])
    #------------------------------------Loop Ends---------------------------------------------        
    N2 = N2[['Variable','Good','Bad','Total','Distribution of Good','Distribution of Bad','WoE','IV','Cummulative_IV']]
    N2.index.names = ['Levels']
    N2.to_csv('WoE_IV_Categorical.csv')
    
    fig = plt.figure() 
    plt.figure(figsize = (14,14))

    N2 = pd.DataFrame(N2.groupby('Variable').Cummulative_IV.mean())
    N2 = N2.reset_index()
    plt.title('IV Chart',fontsize = 20)
    plt.barh(N2['Variable'],N2['Cummulative_IV'])


# In[4]:


def Woe_IV_Continuous(data = None, df = None, q = None, TargetVar = None):
    
    """
Find the WoE and IV values of each variable and each unique level of continuous data so as to find out the best predictor. These generated values are stored into a CSV named WoE_IV_Continuous.csv
Following is what the values of IV mean according to Siddiqi (2006):
    •Information Value Predictive Power 
    • < 0.02 useless for prediction 
    •0.02 to 0.1 Weak predictor 
    •0.1 to 0.3 Medium predictor 
    •0.3 to 0.5 Strong predictor 
    • > 0.5 Suspicious or too good to be true
    
    
Parameters
----------
data : imported main dataset on which plots are to be generated 

df : Dataframe of input data report as created as generated using QC type & ip_data_report module

q  : Number of groups/deciles of data we need 

TargetVar : Name of the column of Target Variable in the data.

"""        
    uni=[]
    for i in range(0,len(df)):
        if df['qc_type'][i] == 'uni dist':
            uni.append(df['variable'][i])
    N2 = pd.DataFrame()
#----------------------------------------Loop to perform Woe & Iv tests on every variable---------------------------  
    for x in uni:
        if data[x].dtypes == 'int64' or data[x].dtypes == 'float64' :
            n = pd.DataFrame([data[x],data[TargetVar]])
            n = n.transpose()
            new = n.sort_values([x])
            new.columns = ['x','TargetVar']
            new['counter'] = range(0,len(new))
            new['decile'] = pd.qcut(new['counter'],q, labels=False)
            new1 = new.groupby(['decile']).TargetVar.sum()
            new2 = new.groupby(['decile']).counter.count()
            new3 = new.groupby(['decile']).x.min()
            new4 = new.groupby(['decile']).x.max()
            new5 = list()
            for y in range(0,len(new3)):
                new5.append((new3[y],new4[y]))
            N=pd.concat([new1,new2],axis=1)
            N.columns = ['Good','Total']
            N['Range'] = new5
            N['Bad'] = N['Total']-N['Good']
            N['Distribution of Good']=N['Good']/N['Good'].sum()
            N['Distribution of Bad']=N['Bad']/N['Bad'].sum()
            N['WoE'] = np.log(N['Distribution of Good']/N['Distribution of Bad'])
            N['IV'] = (N['Distribution of Good']-N['Distribution of Bad'])*N['WoE']
            N['Variable'] = [x]*len(N)
            N['Cummulative_IV'] = np.ma.masked_invalid(N['IV']).sum()
            N2 = pd.concat([N2,N])
    #----------------------------Loop Ends-----------------------------------------
    N2 = N2[['Variable','Good','Bad','Total','Distribution of Good','Distribution of Bad','WoE','IV','Cummulative_IV']]
    N2.index.names = ['Levels']
    N2.to_csv('Woe_IV_Continuous.csv')
    
    fig = plt.figure() 
    plt.figure(figsize = (14,14))

    N2 = pd.DataFrame(N2.groupby('Variable').Cummulative_IV.mean())
    N2 = N2.reset_index()
    plt.title('IV Chart',fontsize = 20)
    plt.barh(N2['Variable'],N2['Cummulative_IV'])

