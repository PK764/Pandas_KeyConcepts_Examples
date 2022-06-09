#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


pd.__version__


# In[5]:


pd.show_versions()


# In[6]:


import numpy as np


# In[7]:


data={'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


# In[8]:


df=pd.DataFrame(data,index=labels)


# In[9]:


df


# In[10]:


df.info()


# In[11]:


df.iloc[:4]


# In[12]:


df.iloc[:,:2]


# In[13]:


df.iloc[[3,4,8],:2]


# In[14]:


df


# In[15]:


df[df['visits']>3]


# In[16]:


df[df['age'].isnull()]


# In[17]:


df[(df['animal']=='cat') & ( df['age']<3)]


# In[18]:


df[(df['age']>=2) & ( df['age']<=4)]


# In[19]:


df[df['age'].between(2,4)]


# In[20]:


df.loc['f','age']=1.5


# In[21]:


df['visits'].sum()


# In[22]:


df.groupby('animal')['age'].mean()


# In[23]:


df.loc['k']=['horse',5,3,'yes']


# In[24]:


df


# In[25]:


df.drop('k')


# In[26]:


df['animal'].value_counts()


# In[28]:


df.sort_values(by=['age','visits'],ascending=[False,True])


# In[34]:


df['priority']=df['priority'].map({'yes':True,'no':False})


# In[30]:


df


# In[42]:


df['animal']=df['animal'].replace('snake','python')


# In[49]:


df=pd.DataFrame(data,index=labels)


# In[50]:


df


# In[51]:


df['animal']=df['animal'].replace('snake','python')


# In[52]:


df


# In[53]:


df.pivot_table(index='animal',columns='visits',values='age',aggfunc='mean')


# In[54]:


df=pd.DataFrame({'A':[1,2,2,3,4,5,5,5,6,7,7]})


# In[55]:


df


# In[58]:


df['A'].unique()


# In[59]:


df


# In[60]:


df=pd.DataFrame(np.random.random(size=(5,3)))


# In[61]:


df


# In[62]:


df.sub(df.mean(axis=1),axis=0)


# In[63]:


df=pd.DataFrame(np.random.random(size=(5,10)),columns=list('abcdefghij'))


# In[64]:


df


# In[75]:


df.sum().idxmin()


# In[76]:


df = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))
df


# In[80]:


len(df.drop_duplicates())


# In[81]:


df


# In[82]:


nan = np.nan

data = [[0.04,  nan,  nan, 0.25,  nan, 0.43, 0.71, 0.51,  nan,  nan],
        [ nan,  nan,  nan, 0.04, 0.76,  nan,  nan, 0.67, 0.76, 0.16],
        [ nan,  nan, 0.5 ,  nan, 0.31, 0.4 ,  nan,  nan, 0.24, 0.01],
        [0.49,  nan,  nan, 0.62, 0.73, 0.26, 0.85,  nan,  nan,  nan],
        [ nan,  nan, 0.41,  nan, 0.05,  nan, 0.61,  nan, 0.48, 0.68]]

columns = list('abcdefghij')

df = pd.DataFrame(data, columns=columns)


# In[83]:


df


# In[85]:


(df.isnull().cumsum(axis=1)==3).idxmax(axis=1)
    
    


# In[86]:


df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})


# In[87]:


df


# In[92]:


df.groupby('grps')['vals'].nlargest(3).sum(level=0)


# In[93]:


df = pd.DataFrame(np.random.RandomState(8765).randint(1, 101, size=(100, 2)),columns = ["A", "B"])
df


# In[94]:


df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()


# In[95]:


df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})


# In[96]:


df


# In[99]:


dti=pd.date_range(start='2015-01-01',end='2015-12-31',freq='B')
s=pd.Series(np.random.random(len(dti)),index=dti)
s


# In[100]:


s[s.index.weekday==2].sum()


# In[102]:


s.resample('M').mean()


# In[103]:


s.groupby(pd.Grouper(freq='4M')).idxmax()


# In[104]:


pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')


# In[105]:


df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
df


# In[106]:


df['FlightNumber']=df['FlightNumber'].interpolate().astype(int)


# In[107]:


df


# In[108]:


t=df.From_To.str.split('_',expand=True)
t.columns=['From','To']
t


# In[110]:


t['From']=t['From'].str.capitalize()
t['To']=t['To'].str.capitalize()
t


# In[114]:


df=df.drop('From_To',axis=1)


# In[115]:


df


# In[118]:


df=df.join(t,how='left')


# In[119]:


df


# In[121]:


df=df.drop(['From','To'],axis=1)


# In[122]:


df


# In[124]:


df=df.join(t,how='right')
df


# In[125]:


df=df.drop(['From','To'],axis=1)


# In[132]:


df


# In[133]:


df=df.join(t)


# In[134]:


df


# In[137]:


df['Airline']=df['Airline'].str.extract('([a-zA-Z\s]+)',expand=False).str.strip()


# In[136]:


df


# In[138]:


letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)
s


# In[139]:


s.index.is_lexsorted()


# In[140]:


s.loc[:,[1,3,6]]


# In[141]:


s.loc[pd.IndexSlice[:'B', 5:]]


# In[142]:


s.sum(level=0)


# # Minesweeper game

# In[145]:


X=5
Y=4
p = pd.core.reshape.util.cartesian_product([np.arange(X), np.arange(Y)])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])
df


# In[146]:


df['mine'] = np.random.binomial(1, 0.4, X*Y)
df


# In[147]:


df['adjacent'] =     df.merge(df + [ 1,  1, 0], on=['x', 'y'], how='left')      .merge(df + [ 1, -1, 0], on=['x', 'y'], how='left')      .merge(df + [-1,  1, 0], on=['x', 'y'], how='left')      .merge(df + [-1, -1, 0], on=['x', 'y'], how='left')      .merge(df + [ 1,  0, 0], on=['x', 'y'], how='left')      .merge(df + [-1,  0, 0], on=['x', 'y'], how='left')      .merge(df + [ 0,  1, 0], on=['x', 'y'], how='left')      .merge(df + [ 0, -1, 0], on=['x', 'y'], how='left')       .iloc[:, 3:]        .sum(axis=1)


# In[148]:


from scipy.signal import convolve2d

mine_grid = df.pivot_table(columns='x', index='y', values='mine')
counts = convolve2d(mine_grid.astype(complex), np.ones((3, 3)), mode='same').real.astype(int)
df['adjacent'] = (counts - mine_grid).ravel('F')


# In[149]:


df.loc[df['mine'] == 1, 'adjacent'] = np.nan


# In[150]:


df.drop('mine', axis=1).set_index(['y', 'x']).unstack()


# In[ ]:




