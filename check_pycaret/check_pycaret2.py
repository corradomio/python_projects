#
# https://www.datacamp.com/tutorial/guide-for-automating-ml-workflows-using-pycaret
#
import numpy as np

# load the dataset from pycaret
from pycaret.datasets import get_data

data = get_data('diamond')

# plot scatter carat_weight and Price
import plotly.express as px

fig = px.scatter(x=data['Carat Weight'], y=data['Price'], facet_col=data['Cut'], opacity=0.25, trendline='ols',
                 trendline_color_override='red')
fig.show()

# plot histogram
fig = px.histogram(data, x=["Price"])
fig.show()

# plot histogram
data['logged_Price'] = np.log(data['Price'])
fig = px.histogram(data, x=["logged_Price"])
fig.show()

# initialize setup
from pycaret.regression import *

s = setup(data, target='Price', transform_target=True, log_experiment=True, experiment_name='diamond')

# compare all models
best = compare_models()

# check the final params of best model
best.get_params()

# check the residuals of trained model
plot_model(best, plot='residuals_interactive')

plot_model(best, plot='feature')

evaluate_model(best)


# copy data and remove target variable
data_unseen = data.copy()
data_unseen.drop('Price', axis = 1, inplace = True)
predictions = predict_model(best, data = data_unseen)

save_model(best, 'my_best_pipeline')

