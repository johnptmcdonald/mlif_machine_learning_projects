import pandas as pd
import joblib as joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./house_data.csv')
X = df[['sq_feet', 'num_bedrooms', 'num_bathrooms']]
y = df['sale_price']

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.25
)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'house_value_model.pkl')

# -----------------------------
print('Model training results')

mae_train = mean_absolute_error(
	y_train, 
	model.predict(X_train)
)

print(f' - Training set error: {mae_train}')

mae_test = mean_absolute_error(
	y_test,
	model.predict(X_test)
)

print(f' - Test set error: {mae_test}')