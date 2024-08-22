
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# #### 2.2. Loading The Dataset <a id =7 ><a/>

# In[2]:


df = pd.read_csv('heart.csv')
df.head()


# #### 2.3. Initial analysis on the dataset <a id =8 ><a/> 

# In[3]:


new_colums = ["age", "sex", "cp", "trtbps", "chol", "fbs", "rest_ecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]


# In[4]:


df.columns = new_colums


# In[5]:


df


# In[6]:


print("Shape of the Dataset", df.shape)


# #### 2.3.1. Analysis Outputs <a id =9 ><a/>

# In[7]:


df.info()


# ### 3. Preparation for Exploaratory Data Analysis(EDA) <a id =10 ><a/> 

# #### * 3.1. Examining Missing Values <a id =11 ><a/> 

# In[8]:


df.isnull().sum()


# In[9]:


isnull_number = []
for i in df.columns: 
    x = df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[10]:


import missingno
missingno.bar(df, color = 'b')


# ### 3.2. Examining Uniques Values <a id =12 ><a/>

# In[11]:


df.head()


# In[12]:


df['cp'].value_counts()


# In[13]:


df['cp'].value_counts().count()


# In[14]:


unique_number = []
for i in df.columns: 
    x = df[i].value_counts().count()
    unique_number.append(x)
pd.DataFrame(unique_number, index = df.columns, columns = ["Total Uniques Values"])



# In[15]:


df.head


# In[16]:


numeric_var = ["age", "trtbps", "chol", "thalach", "oldpeak"]
categoric_var = ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal", "target"]


# In[17]:


df[numeric_var].describe()


# In[18]:


sns.distplot(df["age"], hist_kws= dict(linewidth=1, edgecolor = "k"))


# In[19]:


sns.distplot(df["trtbps"], hist_kws= dict(linewidth=1, edgecolor = "k"), bins = 20)


# In[20]:


sns.distplot(df["chol"], hist= False)


# In[21]:


x,y = plt.subplots(figsize = (8, 6))
sns.distplot(df["thalach"], hist= False, ax = y)
y.axvline(df["thalach"].min(), color ="r", ls="--")
y.axvline(df["thalach"].mean(), color ="r", ls="--")


# In[22]:


x,y = plt.subplots(figsize = (8, 6))
sns.distplot(df["oldpeak"], hist_kws= dict(linewidth=1, edgecolor = "k"), bins = 20, ax = y)
# sns.distplot(df["oldpeak"], hist= False, ax = y), bins = 20)
y.axvline(df["oldpeak"].mean(), color ="r", ls="--")



# ### 4.Exploratory Data Analysis(EDA) <a id =15 ><a/> 

# In[23]:


numeric_var


# In[24]:


numerix_axis_name = ["Age of the Patient", "Resting Blood Pressure", "Cholesterol", "Maximum Heart Rate Achieved", "ST Depression"]


# In[25]:


list(zip(numeric_var, numerix_axis_name))


# In[26]:


title_font = {"family": "arial", "color":"darkred", "weight": "bold", "size": 15}
axis_font = {"family": "arial", "color":"darkblue", "weight": "bold", "size": 13}

for i, z in list(zip(numeric_var, numerix_axis_name)): 
    plt.figure(figsize=(8, 6), dpi = 80)
    sns.distplot(df[i], hist_kws= dict(linewidth=1, edgecolor = "k"), bins = 20)
    
    plt.title(z, fontdict=title_font)
    plt.xlabel(z, fontdict=axis_font)
    plt.ylabel("Density", fontdict=axis_font)
    
    plt.tight_layout()
    plt.show()
    


# In[27]:


categoric_var


# In[28]:


categoric_axis_name = ["Gender", "Chest Pain Type", "Fasting Blood Sugar", "Resting Electrocardiographic Results", "Exercise Induced Angina", "The Slope of ST Segment", "Number of Major Vessels", "Thal", "Target"]


# In[29]:


list(zip(categoric_var, categoric_axis_name))


# In[30]:


df["cp"].value_counts()


# In[31]:


list(df["cp"].value_counts().index)


# In[32]:


title_font = {"family": "arial", "color":"darkred", "weight": "bold", "size": 15}
axis_font = {"family": "arial", "color":"darkblue", "weight": "bold", "size": 13}

for i, z in list(zip(categoric_var, categoric_axis_name)): 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    observation_values = list(df[i].value_counts().index)
    total_observation = list(df[i].value_counts())
    ax.pie(total_observation, labels=observation_values, autopct="%1.1f%%", startangle = 110, labeldistance = 1.1)
    ax.axis("equal")
    plt.title((i + "(" + z + ")"), fontdict = title_font)
    plt.legend()
    plt.show()
    
    


# #### 4.1.2.2. Examining the Missing Data According to the Analysis Result <a id =18 ><a/> 

# In[33]:


df[df['thal'] == 0]


# In[34]:


df['thal'] = df['thal'].replace(0, np.nan)


# In[35]:


df.loc[[48,281], :]


# In[36]:


isnull_number = []
for i in df.columns: 
    x = df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[37]:


df['thal'].fillna(2, inplace = True)


# In[38]:


df.loc[[48,281], :]


# In[39]:


df


# In[40]:


df['thal'] = pd.to_numeric(df['thal'], downcast = 'integer')


# In[41]:


df.loc[[48,281], :]


# In[42]:


isnull_number = []
for i in df.columns: 
    x = df[i].isnull().sum()
    isnull_number.append(x)
pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[43]:


df['thal'].value_counts()


# 4.2. Bi-variate Analysis 
#     

# 4.2.1. Numerical Variables - Target Variable (Analysis with FaceGrid)

# In[44]:


numeric_var.append('target')


# In[45]:


numeric_var


# In[46]:


title_font = {"family": "arial", "color":"darkred", "weight": "bold", "size": 15}
axis_font = {"family": "arial", "color":"darkblue", "weight": "bold", "size": 13}

for i, z in list(zip(numeric_var, numerix_axis_name)): 
    graph = sns.FacetGrid(df[numeric_var], hue = 'target', height =5, xlim = ((df[i].min()- 10), (df[i].max() + 10)))
    graph.map(sns.kdeplot, i, shade = True)
    graph.add_legend()

    
    plt.title(z, fontdict=title_font)
    plt.xlabel(z, fontdict=axis_font)
    plt.ylabel("Density", fontdict=axis_font)
    
    plt.tight_layout()
    plt.show()


# In[47]:


df[numeric_var].corr()


# In[48]:


df[numeric_var].corr().iloc[:, [-1]]


# 4.2.2. Categorical Variables - Target variables (Analysis with Count Plot)

# In[49]:


# for i, z in list(zip(categoric_var, categoric_axis_name)): 
    
#     plt.figure(figire = (8, 5))
#     sns.countplot(i, data = df[categoric_var], hue = 'target')

    
#     plt.title(i + ' - target', fontdict=title_font)
#     plt.xlabel(z, fontdict=axis_font)
#     plt.ylabel("Target", fontdict=axis_font)
    
#     plt.tight_layout()
#     plt.show()

for i, z in list(zip(categoric_var, categoric_axis_name)): 
    plt.figure(figsize=(7, 4))  
    sns.countplot(x=i, data=df, hue='target')  

    plt.title(i + ' - target', fontdict=title_font)
    plt.xlabel(z, fontdict=axis_font)
    plt.ylabel("Target", fontdict=axis_font)

    plt.tight_layout()
    plt.show()


# In[50]:


df[categoric_var].corr()


# In[51]:


df[categoric_var].corr().iloc[:, [-1]]


# 4.2.3. Examining Numeric Variables amond themselves (analysis with pair plot)

# In[52]:


numeric_var


# In[53]:


numeric_var.remove('target')


# In[54]:


numeric_var


# In[55]:


df[numeric_var].head()


# In[56]:


graph = sns.pairplot(df[numeric_var], diag_kind = 'kde')
graph.map_lower(sns.kdeplot, levels = 4, color = '.2')
plt.show()

# The provided code creates a pair plot using the seaborn library to visualize 
# the relationships between numerical variables in a dataset.


# 4.2.4. Feature Scaling wih the RobustScaler Method 

# In[57]:


from sklearn.preprocessing import RobustScaler


# In[58]:


robust_scaler = RobustScaler()


# In[59]:


scaled_data = robust_scaler.fit_transform(df[numeric_var])


# In[60]:


scaled_data


# In[61]:


type(scaled_data)


# In[62]:


df_scaled = pd.DataFrame(scaled_data, columns = numeric_var)
df_scaled.head()


# 4.2.5. Creating a New DataFrame with the Melt() Function 

# In[63]:


df_new = pd.concat([df_scaled, df.loc[:, 'target']], axis = 1)


# In[64]:


df_new


# In[65]:


# using the Melt function : allows us to create a pivot table 
melted_data = pd.melt(df_new, id_vars = 'target', var_name = 'variables', value_name = 'value')


# In[66]:


melted_data


# In[67]:


plt.figure(figsize = (8, 5))
sns.swarmplot(x= 'variables', y = 'value', hue ='target', data = melted_data)
plt.show()


# 4.2.6. Numerical variables  - Categorical Variables (Analysis with Swarm Plot)

# In[68]:


axis_font = {"family": "arial", "color":"black", "weight": "bold", "size": 14}

for i in df[categoric_var]: 
    df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
    melted_data = pd.melt(df_new, id_vars = i, var_name = 'variables', value_name = 'value')
    
    plt.figure(figsize = (8, 5))
    sns.swarmplot(x= 'variables', y = 'value', hue =i, data = melted_data)
    
    plt.xlabel('variables', fontdict = axis_font)
    plt.ylabel('value', fontdict = axis_font)
    
    plt.tight_layout()
    
    plt.show()


# 4.2.7. Numerical Variables - Categorical Variables (Analysis with Box Plot )

# In[69]:


axis_font = {"family": "arial", "color":"black", "weight": "bold", "size": 14}

for i in df[categoric_var]: 
    df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
    melted_data = pd.melt(df_new, id_vars = i, var_name = 'variables', value_name = 'value')
    
    plt.figure(figsize = (8, 5))
    sns.boxplot(x= 'variables', y = 'value', hue =i, data = melted_data)
    
    plt.xlabel('variables', fontdict = axis_font)
    plt.ylabel('value', fontdict = axis_font)
    
    plt.tight_layout()
    
    plt.show() 


# 4.2.8. Numerical - Categorical Variables (Analysis with the Heatmap)

# In[70]:


df_scaled


# In[71]:


df_new2 = pd.concat([df_scaled, df[categoric_var]], axis =1)


# In[72]:


categoric_var


# In[73]:


df_new2.corr()


# In[74]:


plt.figure(figsize = (15, 10))
sns.heatmap(data = df_new2.corr(), cmap = 'Spectral', annot = True, linewidths=0.5)


# 5. Preparation for Modeling 

# In[75]:


df.head()


# In[76]:


df.drop(['chol', 'fbs', 'rest_ecg'], axis = 1, inplace = True)


# In[77]:


df.head()


# 5.2. Struggling Outlisers 

# 5.2.1. Visualizing outliers 

# In[78]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

ax1.boxplot(df['age'])
ax1.set_title('age')

ax2.boxplot(df['trtbps'])
ax2.set_title('trtbps')

ax3.boxplot(df['thalach'])
ax3.set_title('thalach')

ax4.boxplot(df['oldpeak'])
ax4.set_title('oldpeak')

plt.show()




# 5.2.2. Dealing with outliers 

# 5.2.2.1. Trtbps Variable 

# In[79]:


from scipy import stats 
from scipy.stats import zscore 
from scipy.stats.mstats import winsorize 


# In[80]:


# we can determnine the number of outliers above the threshold
z_scores_trtbps = zscore(df['trtbps'])
for threshold in range(1, 4): 
    print('Threshold Value:  {}'.format(threshold))
    print('Number of Outliers: {}'.format(len(np.where(z_scores_trtbps    > threshold)[0])))
    print('-------------------')


# In[81]:


df[z_scores_trtbps > 2][['trtbps']]


# In[82]:


df[z_scores_trtbps > 2].trtbps.min()


# In[83]:


df[z_scores_trtbps > 2].trtbps.max()


# In[84]:


df[df['trtbps'] < 170].trtbps.max()


# In[85]:


winsorize_percentile_trtbps = (stats.percentileofscore(df['trtbps'], 165)) / 100 
print(winsorize_percentile_trtbps)


# In[86]:


1 - winsorize_percentile_trtbps


# In[87]:


trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))


# In[88]:


trtbps_winsorize


# In[89]:


plt.boxplot(trtbps_winsorize)
plt.xlabel('trtbps_winsorize', color = 'b')
plt.show()


# In[90]:


df['trtbps_winsorize'] = trtbps_winsorize


# In[91]:


df.head()


# 5.2.2. Thalach Variables 

# In[92]:


def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)    
    diff = q3 - q1 
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]


# In[93]:


thalach_out = iqr(df, 'thalach')
thalach_out


# In[94]:


df.drop([272], axis =0, inplace=True)


# In[95]:


df['thalach'][270:275]


# In[96]:


plt.boxplot(df['thalach'])


# 5.2.2.3. Pldpeak Variable

# In[97]:


iqr(df, 'oldpeak')


# since we have 5 outliers, we dont want to remvove them from the dataset, we will apply to winriser

# In[98]:


df[df['oldpeak'] < 4.2].oldpeak.max()


# In[99]:


winsorize_percentile_oldpeak = stats.percentileofscore(df['oldpeak'], 4)/100 
print(winsorize_percentile_oldpeak)


# In[100]:


1- winsorize_percentile_oldpeak


# In[101]:


oldpeak_winsorize = winsorize(df.oldpeak, (0, (1 - winsorize_percentile_oldpeak)))
oldpeak_winsorize


# In[102]:


plt.boxplot(oldpeak_winsorize)
plt.xlabel('oldpeak_winsorize', color = 'b')
plt.show()


# In[103]:


df['oldpeak_winsorize'] = oldpeak_winsorize


# In[104]:


df.head()


# In[105]:


df.drop(['trtbps', 'oldpeak'], axis=1, inplace=True)
df.head()


# 5.3. Determing Distributions of Numerica Variables 

# In[106]:


df.head()


# In[107]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

ax1.hist(df['age'])
ax1.set_title('age')

ax2.hist(df['trtbps_winsorize'])
ax2.set_title('trtbps_winsorize')

ax3.hist(df['thalach'])
ax3.set_title('thalach')

ax4.hist(df['oldpeak_winsorize'])
ax4.set_title('oldpeak_winsorize')

plt.show()




# In[108]:


df[['age', 'trtbps_winsorize', 'thalach', 'oldpeak_winsorize']].agg(['skew']).transpose()



# 5.4. Transformation Operations on Unsymmetrical Data 

# In[109]:


df['oldpeak_winsorize_log'] = np.log(df['oldpeak_winsorize'])
df['oldpeak_winsorize_sqrt'] = np.sqrt(df['oldpeak_winsorize'])


# In[110]:


df.head()


# In[111]:


df[['oldpeak_winsorize', 'oldpeak_winsorize_log', 'oldpeak_winsorize_sqrt']].agg(['skew']).transpose()


# In[112]:


df.drop(['oldpeak_winsorize', 'oldpeak_winsorize_log'], axis =1, inplace = True)


# In[113]:


df.head()


# 5.5. Applyign one Hot Encoding method to Categorical Variables 

# In[114]:


# one harded coding method - converting categorical variables to bianary 

df_copy = df.copy()


# In[115]:


df_copy 


# In[116]:


categoric_var


# In[117]:


categoric_var.remove('fbs')
categoric_var.remove('rest_ecg')


# In[118]:


df_copy = pd.get_dummies(df_copy, columns= categoric_var[:-1], drop_first=True)


# In[119]:


df_copy.head()


# 5.6. Feature Scaling with the RobuistScaler Method for Machine Learning Algorithms 

# In[120]:


new_numeric_var = ['age', 'thalach', 'trtbps_winsorize', 'oldpeak_winsorize_sqrt']


# In[121]:


new_numeric_var


# In[122]:


robus_scaler = RobustScaler()


# In[123]:


robus_scaler


# In[124]:


df_copy[new_numeric_var] = robus_scaler.fit_transform(df_copy[new_numeric_var])


# In[125]:


df_copy.head()


# 5.7. Separating Data into Test and Training Set 

# In[126]:


# from sklearn.model_selection import train_test_split 


# In[127]:


# x = df_copy.drop(['target'], axis = 1)
# y = df_copy[['target']]


# In[128]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= = 0.1, random_state = 3)


# In[129]:


from sklearn.model_selection import train_test_split

x = df_copy.drop(['target'], axis=1)
y = df_copy[['target']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3)


# In[130]:


X_train.head()


# In[131]:


y_train.head()


# In[132]:


print(f'X_train:{X_train.shape[0]}')
print(f'X_test:{X_test.shape[0]}')
print(f'y_train:{y_train.shape[0]}')
print(f'y_test:{y_test.shape[0]}')


# 6. Modeling 

# 6.1. Logistic Regression Algorithm 

# In[133]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[134]:


log_reg = LogisticRegression()


# In[135]:


log_reg


# In[136]:


log_reg.fit(X_train, y_train)


# In[137]:


y_pred = log_reg.predict(X_test)


# In[138]:


y_pred


# In[139]:


accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy: {}'.format(accuracy))


# 6.1.1. Crossa Validation 

# In[140]:


from sklearn.model_selection import cross_val_score


# In[141]:


scores = cross_val_score(log_reg, X_test, y_test, cv = 10)
print("Cross Validtion Accuracy Scores", scores.mean())


# ![image.png](attachment:image.png)

# In[142]:


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(log_reg, X_test, y_test, name = 'Logistic Regression')
plt.title('Logistic Regression ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# 6.1.3. Hyperparameter Optimization (with GridSearchCV)

# In[143]:


from sklearn.model_selection import GridSearchCV


# In[144]:


log_reg_new = LogisticRegression()


# In[145]:


log_reg_new


# In[146]:


parameters = {'penalty' : ['l1', 'l2'], 'solver' : ['liblinear', 'sage', 'sage']}


# In[147]:


log_reg_grid = GridSearchCV(log_reg_new, param_grid = parameters)


# In[148]:


log_reg_grid.fit(X_train, y_train)


# In[149]:


print('Best Parameters:', log_reg_grid.best_params_)


# In[150]:


log_reg_new2 = LogisticRegression(penalty='l1', solver='liblinear')
log_reg_new2


# In[151]:


log_reg_new2.fit(X_train, y_train)


# In[152]:


y_pred = log_reg_new2.predict(X_test)


# In[153]:


print('The test accuracty score of Logistic Regression After hyper-parameter tuning is : {}'.format(accuracy_score(y_test, y_pred)))


# In[154]:


RocCurveDisplay.from_estimator(log_reg_new2, X_test, y_test, name = 'Logistic GridSearchCV')
plt.title('Logistic Regression GridSearchCV ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# 6.2. Decision Tree Algorithm 

# In[155]:


from sklearn.tree import DecisionTreeClassifier


# In[156]:


dec_tree = DecisionTreeClassifier(random_state= 5)


# In[157]:


dec_tree.fit(X_train, y_train)


# In[158]:


y_pred = dec_tree.predict(X_test)


# In[159]:


print('The test accuracy score of Decision Tree is : ', accuracy_score(y_test, y_pred))


# In[160]:


scores = cross_val_score(dec_tree, X_test, y_test, cv = 10)
print("Cross Validtion Accuracy Scores", scores.mean())


# In[161]:


RocCurveDisplay.from_estimator(dec_tree, X_test, y_test, name = 'Decision Tree')
plt.title('Logistic Regression Decision Tree ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# 6.3. Support Vector Machine Algorithm

# In[162]:


from sklearn.svm import SVC


# In[163]:


svc_model = SVC(random_state =5)


# In[164]:


svc_model.fit(X_train, y_train)


# In[165]:


y_pred = svc_model.predict(X_test)


# In[166]:


print('The test accuracty score of SVM is:', accuracy_score(y_test, y_pred))


# In[167]:


scores = cross_val_score(svc_model, X_test, y_test, cv = 10)
print("Cross Validtion Accuracy Scores", scores.mean())


# In[168]:


RocCurveDisplay.from_estimator(svc_model, X_test, y_test, name = 'Support Vector Machine')
plt.title('Logistic Regression Support Vector Machine ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# 6.4. Random Forest Algorithm 

# In[169]:


from sklearn.ensemble import RandomForestClassifier


# In[170]:


random_forest = RandomForestClassifier(random_state=5)


# In[171]:


random_forest.fit(X_train, y_train)


# In[172]:


y_pred = random_forest.predict(X_test)


# In[173]:


print('The test accuracty score of Random Forest is', accuracy_score(y_test, y_pred))


# In[174]:


scores = cross_val_score(random_forest, X_test, y_test, cv = 10)
print("Cross Validtion Accuracy Scores", scores.mean())


# In[175]:


RocCurveDisplay.from_estimator(random_forest, X_test, y_test, name = 'Random Forest')
plt.title('Logistic Regression Random Forest ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# 6.4.1. Hyperparameter Optimization (with GirdSearchCV)

# In[176]:


random_forest_new = RandomForestClassifier(random_state = 5)
random_forest_new


# In[177]:


parameters = {'n_estimators':  [50, 100, 150, 200], 
             'criterion': ['gini', 'entropy'], 
             'max_features': ['auto', 'sqrt', 'log2'], 
             'bootstrap': [True, False]}


# In[178]:


random_forest_grid = GridSearchCV(random_forest_new, parameters)


# In[ ]:


random_forest_grid.fit(X_train, y_train)


# In[ ]:


print('Best Parameters:', random_forest_grid.best_params_)


# In[ ]:


random_forest_new2 = RandomForestClassifier(bootstrap=True, criterion = 'entropy', max_features='auto', n_estimators=200, random_state=5)


# In[ ]:


# random_forest_new2.fit(X_train, y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest_new2 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features=None, n_estimators=200, random_state=5)
random_forest_new2.fit(X_train, y_train)


# In[ ]:


y_pred = random_forest_new2.predict(X_test)


# In[ ]:


print('The test accuracy of Random Forest after hyper-parameter tuning is:', accuracy_score(y_test, y_pred))


# In[ ]:


RocCurveDisplay.from_estimator(random_forest_new2, X_test, y_test, name = 'Random Forest')
plt.title('Random Forest ROC Curve and AUC')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# In[ ]:


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # # Load the dataset
# # url = 'https://raw.githubusercontent.com/dataprofessor/data/master/heart.csv'
# # df = pd.read_csv(url)

# # Define features (X) and target (y)
# X = df.drop(columns=['output'])  # Features
# y = df['output']  # Target variable

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize Random Forest Classifier
# clf = RandomForestClassifier(random_state=42)

# # Train the model
# clf.fit(X_train, y_train)

# # Predictions
# y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # Classification report
# print(classification_report(y_test, y_pred))


# In[ ]:


X_test


# In[ ]:


X_test.shape


# In[ ]:


X_test.iloc[30]


# In[ ]:





# In[ ]:





# In[ ]:




