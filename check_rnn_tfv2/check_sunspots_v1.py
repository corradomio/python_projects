import pandas as pd
import matplotlib.pyplot as plt

print("Hello World")

# ,Year,Month,Day,Date In Fraction Of Year,Number of Sunspots,Standard Deviation,Observations,Indicator

df = pd.read_csv("sunspot_data.csv")
data = df["Number of Sunspots"]

plt.clf()
plt.plot(data)
plt.title("Number of Sunspots")
plt.show()

X = df.values
diff = list()
cycle = 132
for i in range(cycle, len(X)):
    value = X[i] - X[i - cycle]
    diff.append(value)

plt.clf()
plt.plot(diff)
plt.title('Sunspots Dataset Differences')
plt.show()


import numpy as np
from statsmodels.tsa.arima_model import ARIMA
y = np.array(RNN_data.value)
model = ARIMA(y, order=(1,0,1)) #ARMA(1,1) model
model_fit = model.fit(disp = 0)
print(model_fit.summary())
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
# Actual vs Fitted
cut_t = 30
predictions = model_fit.predict()
plot = pd.DataFrame({'Date':date,'Actual':abs(y[cut_t:]),"Predicted": predictions[cut_t:]})
plot.plot(x='Date',y=['Actual','Predicted'],title = 'ARMA(1,1) Sunspots Prediction',legend = True)
RMSE = np.sqrt(np.mean(residuals**2))


if __name__ == "__main__":
    # main()
    pass
