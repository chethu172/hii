#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
c=[1,2,3,54,6]
sum1=np.sum(c)
print("sum:",sum1)
avg=np.median(c)
print("avg:",avg)
std=np.std(c)
print("std:",std)
max1=np.max(c)
print("max:",max1)
min1=np.min(c)
print("min:",min1)
csum=np.cumsum(c)
print("cssum:",csum)
csprod=np.cumprod(c)
print("csprod:",csprod)
argmin=np.argmin(c)
print("argmin:",argmin)
argmax=np.argmax(c)
print("argmax:",argmax)
corre=np.corrcoef(c)
print("correlation:",corre)


# In[2]:


lambda_cube=lambda y:y*y*y
lambda_cube(5)


# In[3]:


from functools import reduce
def sum1(x,y):
    return x+y
l1=[1,2,3,4]
l2=(reduce(sum1,l1))
l2


# In[4]:


def oddeven(x):
    if x%2==0:
        return True
    else:
        return False
l1=[1,2,3,4,5,6]
l2=list(filter(oddeven,l1))
l2


# In[5]:


def add4(x):
    return x+4
l1=[1,2,3,4,5]
l2=list(map(add4,l1))
l2


# In[6]:


import pandas as pd 
df=pd.read_csv("customer_id.csv")
df


# In[7]:


df.plot("Age","Annaual_income",kind="scatter",marker="o")


# In[8]:


df["Spending_score(1-100)"].plot(kind="bar")


# In[9]:


df["Spending_score(1-100)"].plot(kind="box")


# In[10]:


df["Spending_score(1-100)"].plot(kind="hist")


# In[11]:


import pandas as pd 
data=pd.DataFrame({'value':[1,2,3,45,2,13,98]})
mean=np.mean(data['value'])
std=np.std(data['value'])
print("mnean:",mean)
print("std:",std)
threshold=2 
outliers=[]
for i in data['value']:
    z=(i-mean)/std
    if z>threshold:
        outliers.append(i)
print("outliers:",outliers)


# In[12]:


q1=data['value'].quantile(0.25)
q3=data['value'].quantile(0.75)
IQR=q3-q1
lowerbound=q1-1.5*IQR
upperbound=q3+1.5*IQR
outliers=data[(data['value']<lowerbound) | (data['value']>upperbound)]
print(outliers)


# In[13]:


m=np.mean(data['value'])
print("mean:",m)
for i in data['value']:
    if i <lowerbound or i >upperbound:
        data['value']=data['value'].replace(i,m)


# In[14]:


m=(data['value'].median())
print("mean:",m)
for i in data['value']:
    if i <lowerbound or i >upperbound:
        data['value']=data['value'].replace(i,m)


# In[15]:


for i in data['value']:
    if i <lowerbound or i >upperbound:
        data['value']=data['value'].replace(i,0)


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data['value'])
plt.title("boxplot")
plt.show()


# In[17]:


plt.scatter(data.index,data['value'])
plt.title("scatter plot")
plt.xlabel("index")
plt.ylabel("value")
plt.show()


# In[18]:


plt.hist(data['value'])
plt.title("histogram")
plt.xlabel('value')
plt.ylabel('Frequency')
plt.show()


# In[19]:


sns.histplot(data['value'])
plt.title("histplot")
plt.xlabel('value')
plt.ylabel('density')
plt.show()


# In[20]:


import pandas as pd 
data=({'BP':[12,34,56,78,12],
      'SUGUR':[890,34,123,456,765],
      'AGE':[18,23,54,87,90],
      'HEART DEASISES':[0,1,1,1,0]})
df=pd.DataFrame(data)
df


# In[21]:


df['total']=[1,3,2,5,6] 
df


# In[22]:


df.loc[5]=[1,3,5,7,1]
df


# In[23]:


df.drop(index=5,axis=0)


# In[24]:


df.drop(columns="total",axis=1)


# In[25]:


df['percentage']=np.mean(df,axis=1)


# In[26]:


df


# In[27]:


df+5


# In[28]:


df+[3,6,2,2,4,8]


# In[29]:


df


# In[30]:


df.info()


# In[31]:


df.describe()


# In[32]:


df.size


# In[33]:


df.head()


# In[34]:


df.tail()


# In[35]:


df.shape


# In[36]:


df.mean()


# In[37]:


df.median()


# In[38]:


df.isnull()


# In[39]:


df.notnull()


# In[40]:


df.fillna("bfill")


# In[41]:


df.interpolate()


# In[45]:


from sklearn.datasets import load_iris
df=load_iris()
df


# In[51]:


x=df.data
y=df.target


# In[53]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[54]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[55]:


y_pred=lr.predict(x_test)


# In[65]:


from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
re2=r2_score(y_test,y_pred)
print(re2)


# In[71]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[72]:


y_pred=lr.predict(x_test)


# In[73]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[66]:


from sklearn.svm import SVC
sv=SVC(kernel="linear",gamma=0.5)
sv.fit(x_train,y_train)


# In[69]:


y_pred=sv.predict(x_test)


# In[70]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[74]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[75]:


y_pred=sv.predict(x_test)


# In[76]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[77]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train,y_train)


# In[78]:


y_pred=sv.predict(x_test)


# In[79]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[83]:


from sklearn.ensemble import RandomForestClassifier
dt=RandomForestClassifier()
dt.fit(x_train,y_train)


# In[84]:


y_pred=dt.predict(x_test)


# In[85]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[88]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cf=confusion_matrix(y_pred,y_test)
plt.figure()
sns.heatmap(cf,annot=True)
plt.xlabel("prediction")
plt.ylabel("target")
plt.title("confusion matrix")


# In[89]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[90]:


df=pd.read_csv("customer_id.csv")
df


# In[91]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['Gender']=label_encoder.fit_transform(df['Gender'])
df


# In[92]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No.of Clusters')
plt.ylabel('wcss')
plt.show()


# In[93]:


km1=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km1.fit_predict(x)


# In[94]:


y_means


# In[95]:


km1.cluster_centers_


# In[96]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='pink',label='C1:kanjoos')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='yellow',label='C2:Average')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='cyan',label='C3:Backra/Target')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='magenta',label='C4:Pokiri')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='orange',label='C5:Intelligent')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],s=50,c='blue',label='centeroid')
plt.title("kmeans clusters")
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# In[97]:


file=open("NLP.txt",'r')
text=file.read()
print(text)


# In[98]:


from nltk.tokenize import word_tokenize
words=word_tokenize(text)
print("the total no of words:",len(words))
print(words)


# In[99]:


from nltk.tokenize import sent_tokenize
sentence=sent_tokenize(text)
print("no of sentences:",len(sentence))

for i in range(len(sentence)):
    print("sentence",i+1,":",sentence[i])


# In[100]:


from nltk.probability import FreqDist
all_fdist=FreqDist(words)
all_fdist


# In[101]:


import pandas as pd 
import matplotlib.pyplot as plt 
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(10,10))
all_fdist.plot(kind='bar')
plt.title("frequence distribution")
plt.xlabel("count")
plt.ylabel("words")
plt.show()


# In[102]:


import nltk
words=word_tokenize(text)
stopwords=nltk.corpus.stopwords.words('english')
words_sw_removes=[]
for word in words: 
    if word in stopwords:
        pass
    else:
        words_sw_removes.append(word)


# In[103]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
all_fdist=FreqDist(words_sw_removes).most_common(20)
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(7,7))
all_fdist.plot(kind="bar")
plt.xlabel("words")
plt.ylabel("count")
plt.show()


# In[105]:


from wordcloud import WordCloud,STOPWORDS
import pandas as pd 
import matplotlib.pyplot as plt 
stopwords=set(STOPWORDS)
word_cloud=WordCloud(height=800,width=800,background_color="white",stopwords=stopwords,min_font_size=10).generate(text)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(word_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[106]:


from skimage.io import imread
cloud=imread("cloud.png")
plt.imshow(cloud)


# In[108]:


from wordcloud import WordCloud,STOPWORDS
import pandas as pd 
import matplotlib.pyplot as plt 
stopwords=set(STOPWORDS)
word_cloud=WordCloud(height=800,width=800,background_color="white",stopwords=stopwords,mask=cloud,min_font_size=10).generate(text)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(word_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[109]:


import nltk 
from nltk.metrics.distance import edit_distance
from nltk.corpus import words 
nltk.download('words')
corect_words=words.words()

incorrect_words=['happy','amazaing','intilentg']
for word in incorrect_words:
    temp=[(edit_distance(word,w),w)for w in corect_words if w[0]==word[0]]
    print(sorted(temp,key=lambda val:val[0])[0][1])


# In[110]:


from nltk.tokenize import word_tokenize
file=open("NLP.txt","r")
text=file.read()
text=text.lower()
import re
text=re.sub('[^A-Za-zO9]'," ",text)
text=re.sub("\S\*\d\S"," ",text).strip()
print(text)


# In[111]:


from nltk.stem import PorterStemmer
words=word_tokenize(text)
ps=PorterStemmer()
ps_sent=[ps.stem(words_sent)for words_sent in words]
print(ps_sent)


# In[112]:


words=word_tokenize(text)
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
ps_sent=[lem.lemmatize(words_sent)for words_sent in words]
print(ps_sent)


# In[113]:


import nltk 
from nltk.tokenize import word_tokenize
nltk.download("averaged_perception_tagger")

text1="I am very hungry but the fridge is empty"
wods=word_tokenize(text1)
print("parts of speech:",nltk.pos_tag(wods))


# In[114]:


from sklearn.feature_extraction.text import CountVectorizer
sentence=['he is a smart boy.she is also smart',
         'chirag is a smart person']
cv=CountVectorizer()
x=cv.fit_transform(sentence)
x=x.toarray()
vocabulary=sorted(cv.vocabulary_.keys())
print(vocabulary)
print(x)


# In[115]:


from sklearn.feature_extraction.text import CountVectorizer
sentence=['he is a smart boy.she is also smart',
         'chirag is a smart person']
cv=CountVectorizer(ngram_range=(2,2))
x=cv.fit_transform(sentence)
x=x.toarray()
vocabulary=sorted(cv.vocabulary_.keys())
print(vocabulary)
print(x)


# In[116]:


from sklearn.feature_extraction.text import TfidfVectorizer
sentence=['corana virus is a highly inflectios diseas',
         'corana virus affects older people at the most',
         'older people are higly risk due to this diseas']
tfidf=TfidfVectorizer()
transformed=tfidf.fit_transform(sentence)
import pandas as pd 
df=pd.DataFrame(transformed[0].T.todense(),
               index=tfidf.get_feature_names_out(),
               columns=['TF-IDF'])
df=df.sort_values('TF-IDF',ascending=False)
df


# In[ ]:




