



###### Data Preparation ####

"""Step 1:  read armut_data.csv """


import pandas as pd
pd.set_option("display.max_columns", None)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("armut_data.csv")

"""
Step 2: ServiceID represents a different service for each CategoryID.
 Create a new variable that represents these services by combining ServiceID 
 and CategoryID with an underscore "_".
"""

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

df.head()

""" 
Step 3 : The data set consists of the date and time when the service transactions were made, 
and there is no basket definition (e.g., invoice). 
In order to apply Association Rule Learning, a basket definition (e.g., invoice) needs to be created. 
Here, the basket definition refers to the services each customer receives on a monthly basis.
For example, customer with ID 7256 has a basket consisting of services 9_4 and 46_4 that they received in August 2017,
and another basket consisting of services 9_4 and 38_4 that they received in October 2017. 
Baskets should be identified with a unique ID. To do this, first create a new date variable that only includes the year and month. 
Combine the UserID and the newly created date variable with "_" and assign it to a new variable called ID. 
"""

df.info

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

df.info()
df.head()

df["SepetId"] = df["UserId"].astype(str) + "_" + df["New_Date"]



"""Step 1 : Create pivot table """



invoice_product_df = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

"""Step : 2 Create Association Rules """
asr_df = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
asr_rules = association_rules(asr_df, metric="support", min_threshold=0.01)
asr_rules.head()


""" Step 3 : "Using the arl_recommender function,
 provide service recommendations to a user who recently received the service 2_0. """

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(asr_rules, "2_0", 1)
