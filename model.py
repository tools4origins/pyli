import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

np.random.seed(0)
linear_dataset = range(1000000, 2000001)
normal_dataset = np.random.normal(0, 1, 1000000)


dataset = {index: value for index, value in enumerate(sorted(normal_dataset))}
X = np.array([[value] for value in dataset.values()])
y = np.array(list(dataset.keys()))

model = linear_model.LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print(mean_squared_error(y, y_pred))
print("min error: {}".format(min(y_pred-y)))
print("max error: {}".format(max(y_pred-y)))


plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
