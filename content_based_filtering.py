# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:52:25 2021
Master's project: content-based filtering '
@author: xuruizi
"""

import pandas as pd
from scipy.spatial.distance import cosine

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet



# read the data
reviews=pd.read_csv("review_sample.csv")
meta=pd.read_csv("meta_sample.csv")






#get list of possible categories among the products, the output is called cat_list
meta["category"]=meta["category"].astype(str)
meta["asin"]=meta['asin'].astype(str)

#print(meta["category"][:20])
l=[''.join(meta["category"])]

cat_list=""+l[0]
#print(cat_list)
junk='[]'','
for lt in junk:
    cat_list=cat_list.replace(lt, "")

cat_list=cat_list.split("'")

cat_list=set(cat_list)
cat_list=list(cat_list)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")    
print(cat_list)
cat_list=[x for x in cat_list if '/' not in x]

print("a list of all possible categories: ")
cat_list.remove('')
print(cat_list)
#print('Appliances'  in cat_list)



# construct temporary user profile, by merging fictional past rating of a target user with the meta data

past_ratings= [(5,'B001KGXKY6'), (4,'B00VEZFDUC'), (3,'B01GXOYCFU'),(5,'B0053Y2CNG'),(4,'B00XCPWED6'),(2,'B01FG4GJPE')]
rating_df=pd.DataFrame(past_ratings, columns=['rating','asin'])
#print(rating_df)
rating_df=pd.merge(rating_df,meta, on='asin')
#print(rating_df)
rating_df['category'] = rating_df['category'].map(lambda x: x.lstrip('[').rstrip(']'))
rating_df['category'] = rating_df['category'].str.split(',')
#print(rating_df['category'][1])




#create user preferences for each category by adding the ratings of each rated products to that category column in the dataframe cat_score

cat_score=pd.DataFrame(columns=cat_list)
cat_score.loc[0]=0


i=0
for x in rating_df['category']:
    for y in x:
        if str(y).replace("'","").lstrip(" ") in cat_list:
            cat_score[str(y).replace("'","").lstrip(" ")][0]=cat_score[str(y).replace("'","").lstrip(" ")][0]+rating_df['rating'][i]
'''
print('\n')
print("user preferece for each categories based on past ratings")
print(cat_score)        
#print(cat_score['Washer Parts & Accessories'])
'''
# produce score list for each categories from the dataframe cat_score, matching with the list if categories in order
cat_score_list=cat_score.values.tolist()

cat_score_list=cat_score_list[0]
cat_score_list, cat_list=zip(*sorted(zip(cat_score_list,cat_list),reverse=True))
cat_score_list, cat_list=(list(q) for q in zip(*sorted(zip(cat_score_list,cat_list),reverse=True)))

#cat_score_list[0]+=0.01
    



# modify the dataset to add category vector: represented by a string of numbers
# generate score using cosine similarity between the vector for the product and the comparison vector in which all of the values are 1, this can be done because the system does not take negative feedback into account, and the genres are weighted
original_vector=""
comparison_vector=[]
i=0
for x in range(0,len(cat_list)):
    original_vector+="0"
    comparison_vector.append(1)
    i+=1
df_with_vector=meta

df_with_vector["cat_score_vector"]=original_vector
df_with_vector['category'] = df_with_vector['category'].map(lambda x: x.lstrip('[').rstrip(']'))
df_with_vector['category'] = df_with_vector['category'].str.split(',')


#helper function
def list_to_string(s): 
    str1 = "" 
    for ele in s: 
        str1 += str(ele)
    return str1


row_count=0
for x in df_with_vector["cat_score_vector"]:
    index_count=-1
    temp_list=[int(s) for s in x]
    row_value_list= []
    for y in df_with_vector["category"][row_count]:
        row_value_list.append(y.replace("'","").lstrip(" "))
    #print(row_value_list)
    for cat in cat_list:
        index_count+=1
        if cat in row_value_list:
            #print("yay")
            temp_list[index_count]+=1
    df_with_vector["cat_score_vector"][row_count]=list_to_string(temp_list)
    row_count+=1




# use cosine similarity to rank the products based on their affinity to the user profile
df_with_vector["ranking score"]=0.0
weight_list=cat_score_list

row_count2=0
for n in df_with_vector["cat_score_vector"]:
    df_with_vector["ranking score"][row_count2]=cosine(comparison_vector,[int(s) for s in n], weight_list)
    row_count2+=1
#print(df_with_vector["ranking score"])
final_df=df_with_vector.sort_values(by=["ranking score"])

print("final recommendation list in the form of a ranked dataframe: \n")
print(final_df)
print(final_df['ranking score'][450])








