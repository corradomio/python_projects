import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from common import *
from dowhy import CausalModel

df = pd.read_csv('churn-bigml-20.csv')
df = remove_column_spaces(df)
df = df.replace({'International_plan': {'No': False, 'Yes': True}, 'Voice_mail_plan': {'No': False, 'Yes': True}})
df.State = pd.Categorical(df['State'])
df = df[df.columns.difference(['Account_length', 'Area_code', 'Area_code', 'State'])]

graph = nx.DiGraph()
graph.add_nodes_from(df.columns)
graph.add_edges_from(edges(
    ('Voice_mail_plan', 'Number_vmail_messages', 'Churn'),
    ('Customer_service_calls', 'Churn'),
    (['Total_day_calls', 'Total_eve_calls', 'Total_intl_calls', 'Total_night_calls'], 'Churn'),
    ('Total_day_calls', ['Total_day_minutes', 'Total_day_charge']),
    ('Total_eve_calls', ['Total_eve_minutes', 'Total_eve_charge']),
    ('Total_intl_calls', ['Total_intl_minutes', 'Total_intl_charge']),
    ('Total_night_calls', ['Total_night_minutes', 'Total_night_charge']),
    ('International_plan', 'Churn')
))

nx.draw_spring(graph, with_labels=True)
plt.show()
g = to_gml(graph)

print(g)
print(df.info())

# ---------------------------------------------------------------------------

training = df.copy()
causal_graph = g

model = CausalModel(
        data=training,
        graph=causal_graph,
        treatment='Customer_service_calls',
        outcome='Churn')

#Identify the causal effect
estimands = model.identify_effect()
print(estimands)
