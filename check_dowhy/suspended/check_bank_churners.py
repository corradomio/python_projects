from pprint import pprint
from common import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv('../BankChurners.csv')
pprint(df.info())

# Creating the High_Limit attribute
df['High_limit'] = df['Credit_Limit'].apply(lambda x: True if x > 20000 else False)

# Creating True or False columns from the Attrition flag for the churn column
df['Churn'] = df['Attrition_Flag'].apply(lambda x: True if x == 'Attrited Customer' else False)

training = df[['Customer_Age', 'Education_Level', 'Income_Category', 'High_limit', 'Churn']].copy()

#Creating the
# causal_graph = """
#     digraph {
#     High_limit;
#     Churn;
#     Income_Category;
#     Education_Level;
#     Customer_Age;
#     U[label="Unobserved Confounders"];
#
#     Customer_Age -> Education_Level;
#     Customer_Age -> Income_Category;
#     Education_Level -> Income_Category; Income_Category->High_limit;
#
#     U->Income_Category;
#     U->High_limit;
#     U->Churn;
#
#     High_limit->Churn;
#     Income_Category -> Churn;
#     }
#     """

graph = nx.DiGraph()
graph.add_nodes_from(['Customer_Age', 'Education_Level', 'Income_Category', 'High_limit', 'Churn', 'U'])
graph.add_edges_from(edges(
    ['Customer_Age', ['Education_Level', 'Income_Category']],
    ['Education_Level', 'Income_Category', 'High_limit'],
    [['High_limit', 'Income_Category'], 'Churn'],
    ['U', ['Income_Category', 'High_limit', 'Churn']]
))
nx.draw_spring(graph, with_labels=True)
plt.show()
causal_graph = to_gml(graph)
print(causal_graph)


from dowhy import CausalModel
from IPython.display import Image, display

model = CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment='High_limit',
        outcome='Churn')
model.view_model()
# display(Image(filename="causal_model.png"))

# Identify the causal effect
estimands = model.identify_effect()
print(estimands)
