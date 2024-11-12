#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
c=[1,2,3,4,5]
s=np.sum(c)
print("sum:",s)
avg=np.mean(c)
print("avg:",avg)
std=np.std(c)
print("std:",std)
cs=np.cumsum(c)
print("cumsum:",cs)
p=np.prod(c)
print("prod:",p)
cp=np.cumprod(c)
print("cumprod:",cp)
m=np.max(c)
print("max:",m)
mi=np.min(c)
print("min:",mi)
am=np.argmax(c)
print("argmax:",am)
ami=np.argmin(c)
print("argmin:",ami)
cc=np.corrcoef(c)
print("correlation coef:",cc)


# In[2]:


lambda_cube=lambda y:y*y*y
lambda_cube(4)


# In[3]:


def add2(x):
    return x+2 
c=[1,2,3,4,5]
c1=list(map(add2,c))
c1


# In[4]:


def oddeven(x):
    if x%2==0:
        return True
    else:
        return False
c=[1,2,3,4,5]
c1=list(filter(oddeven,c))
c1


# In[5]:


from functools import reduce
def sum(x,y):
    return x+y
c=[1,2,3,4,5]
c1=reduce(sum,c)
c1


# In[6]:


import pandas as pd 
import numpy as np
data=pd.DataFrame({"value":[2,3,4,2,5,4,7,2,6,3,15]})
mean=np.mean(data['value'])
print("mean:",mean)
std=np.std(data['value'])
print("std:",std)
threshold=2
outlier=[]
for i in data['value']:
    z=(i-mean)/std
    if z>threshold:
        outlier.append(i)
print("outliers using z-score method:",outlier)


# In[7]:


q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
IQR=q3-q1
lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR
print("lowerbound:",lower_bound)
print("upperbound:",upper_bound)
outlier=data[(data['value']<lower_bound)|(data['value']>upper_bound)]
print("outliers using IQR method:",outlier)


# In[8]:


data_filtered=data[(data['value']>=lower_bound) & (data['value']<=upper_bound)]
print("deleting outliers:",data_filtered)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data['value'])
plt.title("box plot")
plt.show()


# In[10]:


sns.histplot(data['value'])
plt.title("histplot")
plt.xlabel("density")
plt.ylabel("value")
plt.show()


# In[11]:


plt.hist(data['value'])
plt.title("histogram")
plt.xlabel("frequency")
plt.ylabel("value")
plt.show()


# In[12]:


plt.scatter(data.index,data['value'])
plt.title("scatter plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# In[13]:


import numpy as np 
m=np.mean(data['value'])
print("mean:",m)
for i in data['value']:
    if i <lower_bound or i>upper_bound: 
        data['value']=data['value'].replace(i,m)


# In[14]:


data


# In[15]:


import numpy as np 
m1=(data['value'].median())
print("median:",m1)
for i in data['value']:
    if i <lower_bound or i>upper_bound: 
        data['value']=data['value'].replace(i,m1)


# In[16]:


data


# In[17]:


for i in data['value']:
    if i <lower_bound or i>upper_bound: 
        data['value']=data['value'].replace(i,0)


# In[18]:


import pandas as pd 
import numpy as np
data=({"bp":[1,np.nan,3,45,21],
      "sugur":[2,5,123,176,100],
      "age":[18,1,9,np.nan,21],
     "chlestrol":[12,67,89,123,98],
      "Heart_disease":[0,0,1,1,1]})
dfc=pd.DataFrame(data)
dfc


# In[19]:


dfc['total']=[12,34,56,78,32]
dfc


# In[20]:


dfc.loc[5]=[np.nan,104,25,76,0,34]
dfc


# In[21]:


dfc.drop(columns="total",axis=1,inplace=True)


# In[22]:


dfc


# In[23]:


dfc.drop(index=5,axis=0,inplace=True)


# In[24]:


dfc


# In[25]:


dfc.isnull()


# In[26]:


dfc.notnull()


# In[27]:


dfc.isnull().sum()


# In[28]:


dfc.fillna(23)


# In[29]:


dfc.bfill()


# In[30]:


dfc.ffill()


# In[31]:


dfc.interpolate()


# In[32]:


dfc.mean()


# In[33]:


dfc.median()


# In[34]:


dfc['bp'].fillna(mean)


# In[35]:


dfc.fillna(dfc['age'].median())


# In[36]:


dfc.fillna(21,inplace=True)


# In[37]:


dfc


# In[38]:


import pandas as pd 
from sklearn.datasets import load_breast_cancer


# In[39]:


df1=load_breast_cancer()
df1


# In[40]:


cs=pd.DataFrame(df1.data,columns=df1.feature_names)


# In[41]:


cs


# In[42]:


x=df1.data
y=df1.target


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[45]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[46]:


y_pred=model.predict(x_test)


# In[47]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
acc=r2_score(y_test,y_pred)
print("accuracy:",acc)


# In[48]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[49]:


y_pred=model.predict(x_test)


# In[50]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score,classification_report
acc=accuracy_score(y_test,y_pred)
print("accuracy:",acc)
cl=classification_report(y_test,y_pred)
print("classification report:",cl)


# In[51]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[52]:


y_pred=model.predict(x_test)


# In[53]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
acc=accuracy_score(y_test,y_pred)
print("accuracy:",acc)
cf=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cf)
f1_score=f1_score(y_test,y_pred)
print("f1-score:",f1_score)
recall=recall_score(y_test,y_pred)
print("recall:",recall)
precision=precision_score(y_test,y_pred)
print("precision:",precision)
tn,fp,fn,tp=cf.ravel()
specificity=tn/(tn+fp)
print("specificity:",specificity)


# In[54]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[55]:


y_pred=model.predict(x_test)


# In[56]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
acc=accuracy_score(y_test,y_pred)
print("accuracy:",acc)
cf=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cf)
f1_score=f1_score(y_test,y_pred)
print("f1-score:",f1_score)
recall=recall_score(y_test,y_pred)
print("recall:",recall)
precision=precision_score(y_test,y_pred)
print("precision:",precision)
tn,fp,fn,tp=cf.ravel()
specificity=tn/(tn+fp)
print("specificity:",specificity)


# In[57]:


from sklearn.svm import SVC
model= SVC(kernel="linear",gamma=0.5)
model.fit(x_train,y_train)


# In[58]:


y_pred=model.predict(x_test)


# In[59]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
acc=accuracy_score(y_test,y_pred)
print("accuracy:",acc)
cf=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cf)
f1_score=f1_score(y_test,y_pred)
print("f1-score:",f1_score)
recall=recall_score(y_test,y_pred)
print("recall:",recall)
precision=precision_score(y_test,y_pred)
print("precision:",precision)
tn,fp,fn,tp=cf.ravel()
specificity=tn/(tn+fp)
print("specificity:",specificity)


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier()
model.fit(x_train,y_train)


# In[61]:


y_pred=model.predict(x_test)


# In[62]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
acc=accuracy_score(y_test,y_pred)
print("accuracy:",acc)
cf=confusion_matrix(y_test,y_pred)
print("confusion matrix:",cf)
f1_score=f1_score(y_test,y_pred)
print("f1-score:",f1_score)
recall=recall_score(y_test,y_pred)
print("recall:",recall)
precision=precision_score(y_test,y_pred)
print("precision:",precision)
tn,fp,fn,tp=cf.ravel()
specificity=tn/(tn+fp)
print("specificity:",specificity)


# In[63]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cf=confusion_matrix(y_test,y_pred)
sns.heatmap(cf,annot=True)
plt.title("confusion matrix")
plt.xlabel("predictions")
plt.ylabel("target")
plt.show()


# In[64]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(i)
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()


# In[65]:


km1=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km1.fit_predict(x)


# In[66]:


y_means


# In[67]:


km1.cluster_centers_


# In[68]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='pink',label="c1")
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='yellow',label="c2")
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='red',label="c3")
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='green',label="c4")
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='gray',label="c5")
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],s=100,c="blue",label="centroid")
plt.plot(figsize=(10,10))
plt.title("the kmeans method")
plt.xlabel("count")
plt.ylabel("no of clustres")
plt.legend()
plt.show()


# In[69]:


import nltk


# In[70]:


file=open("NLP.txt")
text=file.read()
print(text)


# In[71]:


from nltk.tokenize import word_tokenize
words=word_tokenize(text)
print("no of words:",len(words))
print(words)


# In[72]:


from nltk.tokenize import sent_tokenize
sentence=sent_tokenize(text)
print("no of sentence:",len(sentence))
for i in range (len(sentence)):
    print("sentence",i+1,":",sentence[i])


# In[73]:


from nltk.probability import FreqDist
all_fdist=FreqDist(words)
all_fdist


# In[74]:


import pandas as pd
import matplotlib.pyplot as plt
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(8,8))
all_fdist.plot(kind="bar")
plt.title("frequeency distribution")
plt.xlabel("count")
plt.ylabel("words")
plt.show()


# In[75]:


import nltk
from nltk.tokenize import word_tokenize
stopwards=nltk.corpus.stopwords.words('english')
words_sw_removed=[]
for word in words:
    if word in stopwards:
        pass
    else:
        words_sw_removed.append(word)


# In[76]:


import pandas as pd
import matplotlib.pyplot as plt
all_fdist=FreqDist(words_sw_removed).most_common(20)
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(8,8))
all_fdist.plot(kind="bar")
plt.title("frequeency distribution")
plt.xlabel("count")
plt.ylabel("words")
plt.show()


# In[77]:


from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt 
stopwards=set(STOPWORDS)
wordcloud=WordCloud(height=800,width=800,background_color="white",stopwords=stopwards,min_font_size=10).generate(text)
plt.plot(figsize=(10,10),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[78]:


from skimage.io import imread
cloud=imread("cloud.png")
plt.imshow(cloud)
plt.show()


# In[79]:


from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt 
stopwards=set(STOPWORDS)
wordcloud=WordCloud(height=800,width=800,background_color="white",stopwords=stopwards,min_font_size=10,mask=cloud).generate(text)
plt.plot(figsize=(10,10),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[80]:


import nltk
from nltk.metrics.distance import edit_distance
from nltk.corpus import words
nltk.download('words')
correct_words=words.words() 

incorrect_words=['happpy','amazaing','intelligetn']
for word in incorrect_words:
    temp=[(edit_distance(word,w),w)for w in correct_words if w[0]==word[0]]
    print(sorted(temp,key=lambda val:val[0])[0][1])


# In[81]:


from nltk.tokenize import word_tokenize
file=open("NLP.txt")
text=file.read()
text.lower()

import re
text=re.sub('[^A-Za-z09]',' ',text)
text=re.sub('S/*/d/S'," ",text).strip()
print(text)


# In[82]:


words=word_tokenize(text)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
ps_sent=[ps.stem(words_sent) for words_sent in words]
print(ps_sent)


# In[83]:


words=word_tokenize(text)
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
lem_sent=[lem.lemmatize(words_sent) for words_sent in words]
print(lem_sent)


# In[84]:


from nltk.tokenize import word_tokenize
nltk.download('averagged_perception_tagger')
text='I am  very hungry but the fridge is empty'
words=word_tokenize(text)
print("parts of speech:",nltk.pos_tag(words))


# In[85]:


from sklearn.feature_extraction.text import CountVectorizer
sentence=['he is a smart boy.she is also a smart',
         'chirag is a smart person']
cv=CountVectorizer(ngram_range=(2,2))
x=cv.fit_transform(sentence)
x=x.toarray()
vocabulary=sorted(cv.vocabulary_.keys())
print(vocabulary)
print(x)


# In[226]:


import pandas as pd 
df=pd.read_csv("mtcars.csv")
df


# In[227]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['model']=label.fit_transform(df['model'])
df['model']


# In[228]:


df


# In[237]:


df.iloc[2]+2


# In[238]:


df


# In[89]:


import missingno as msn
msn.bar(df)


# In[90]:


df.head()


# In[91]:


df.tail()


# In[92]:


df.info()


# In[93]:


df.describe()


# In[94]:


df.shape


# In[95]:


df.size


# In[96]:


df+5 


# In[97]:


df+[3,2,4,6,8,1,2,6,4,2,3,1]


# In[98]:


x=df.drop('carb',axis=1)
y=df['carb']


# In[99]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[100]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)


# In[101]:


y_pred=model.predict(x_test)


# In[102]:


from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,f1_score,recall_score,precision_score
ac=accuracy_score(y_test,y_pred)
print(ac)
cf=confusion_matrix(y_test,y_pred)
print(cf)
f1=f1_score(y_test,y_pred,average="weighted")
print(f1)
recall=recall_score(y_test,y_pred,average="weighted")
print(recall)
presion=precision_score(y_test,y_pred,average="weighted")
print(presion)
#tn,fp,fn,tp=cf.ravel()
#specificity=tn/(tn+fp)
#print(specificity)


# In[103]:


import pandas as pd 
df=pd.read_csv("iris.csv")
df


# In[104]:


import missingno as msn
msn.bar(df)


# In[105]:


import matplotlib.pyplot as plt 
plt.scatter(df['sepallength'],df['sepalwidth'])
plt.xlabel("sepallength")
plt.ylabel("sepalwidth")
plt.title("scatter plot using plt")
plt.show()


# In[106]:


plt.bar(df['petallength'],df['petalwidth'])
plt.title("bar plot using plt")
plt.xlabel("petallength")
plt.ylabel("petalwidth")
plt.show()


# In[107]:


plt.boxplot(df['sepalwidth'])
plt.title("boxplot using plt")
plt.xlabel("sepalength")
plt.ylabel("sepalwidth")
plt.show()


# In[108]:


plt.hist(df['petalwidth'])
plt.title("histogram using plt")
plt.xlabel("petalwidth")
plt.show()


# In[109]:


plt.plot(df['petallength'])
plt.title("line plot using plt")
plt.xlabel("petallength")
plt.show()


# In[110]:


import seaborn as sns
sns.scatterplot(x='sepallength',y='sepalwidth',data=df)
plt.title("scatter plot using sns")
plt.xlabel("sepallength")
plt.ylabel("sepalwidth")
plt.show()


# In[111]:


sns.barplot(x='petallength',y='petalwidth',data=df)
plt.title("barplot using sns")
plt.xlabel("petallength")
plt.ylabel("petalwidth")
plt.show()


# In[112]:


sns.boxplot(x='sepallength',y='sepalwidth',data=df)
plt.title("box plot using sns")
plt.xlabel("sepallength")
plt.ylabel("sepalwidth")
plt.show()


# In[113]:


sns.lineplot(x='petallength',y='petalwidth',data=df)
plt.title("lineplot using sns")
plt.xlabel("petallength")
plt.ylabel("petalwidth")
plt.show()


# In[114]:


sns.histplot(x='sepallength',data=df)
plt.title("hisplot using sns")
plt.xlabel("sepallength")
plt.show()


# In[115]:


sns.displot(x='sepalwidth',data=df)
plt.title("displot using sns")
plt.xlabel("sepalwidth")
plt.show()


# In[116]:


sns.countplot(x='petalwidth',data=df)
plt.title("countplot using sns")
plt.xlabel("petalwidth")
plt.show()


# In[240]:


import pandas as pd 
import numpy as np
data=({'Ename':['chethana','gagana','ammu','bhoomi','teju','pratee'],
      'department':['cs','ec','cs','ec','cs','ec'],
      'experience':[5,4,2,1,6,9],
      'salary':[1000,23100,13340,5423,8791,1239]})
df=pd.DataFrame(data)
df


# In[241]:


#add new column in dataset
df['age']=[19,18,17,18,18,18]
df


# In[242]:


#add new row in daset
df.loc[8]=['bhoomi','cs',1,13421,18]
df


# In[243]:


#df.drop(index=6,axis=0,inplace=True)


# In[244]:


df


# In[245]:


#make a pivot table avg_salary of each employee for each department
avg_sal=df.pivot_table(values='salary',index='department',columns='experience',aggfunc='mean')
avg_sal


# In[246]:


#make a pivot table show sum and mean of salaries of each employee
avg_sal=df.pivot_table(values='salary',index='department',columns='experience',aggfunc=['mean','sum'])
avg_sal


# In[247]:


#unique value in column
df['department'].value_counts()


# In[248]:


#avg salary of employee based on department
avg=df.groupby('department')['salary'].mean()
avg


# In[249]:


#show the each employee salary
avg=df.groupby('Ename')['salary'].mean()
avg


# In[250]:


#highest and lowest salary employee details
high=df[df['salary']==df['salary'].max()]
low=df[df['salary']==df['salary'].min()]
print("high:",high)
print("low:",low)


# In[251]:


h=df['salary'].max() 
low=df['salary'].min()
print("high:",h)
print("low:",low)


# In[252]:


df.columns


# In[253]:


#show the row wise full details
df.iloc[2]


# In[254]:


#show the lessthan 20 age employees
df[df['age']<19]


# In[255]:


#show the greaterthan 19 age employee
df[df['age']>19]


# In[256]:


#show the greaterthan or equalto 19 age employee
df[df['age']>=19]


# In[257]:


#show the salaey equal to salary employee
df[df['salary']==13421]


# In[258]:


#used to show the unique row in dataset
df[df['Ename']=='bhoomi']


# In[259]:


#increment the specific column in dataset
df['age']=df['age']+2


# In[260]:


df


# In[222]:


#increment the specific row in dataset
df.iloc[2]+2


# In[221]:


df


# In[ ]:




