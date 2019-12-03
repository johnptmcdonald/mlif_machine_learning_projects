import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib

# load the dataset
df = pd.read_csv('./house_data.csv')

# create the X and y arrays
X = df[['sq_feet', 'num_bedrooms', 'num_bathrooms']]
y = df[['sale_price']]

# scaler X and y arrays and convert to np arrays
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

X[X.columns] = X_scaler.fit_transform(X[X.columns])
y[y.columns] = y_scaler.fit_transform(y[y.columns])

X = X.to_numpy()
y = y.to_numpy()

# get train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# define model
model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='SGD')

model.fit(
	X_train,
	y_train,
	epochs=50,
	batch_size=8,
	shuffle=True,
	verbose=2
)

joblib.dump(X_scaler, 'X_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

model.save('house_value_model.h5')

predictions_train = model.predict(X_train, verbose=0)
mae_train = mean_absolute_error(
	y_scaler.inverse_transform(predictions_train),
	y_scaler.inverse_transform(y_train)
)
print(f'Training Set Error: {mae_train}')

predictions_test = model.predict(X_test, verbose=0)
mae_test = mean_absolute_error(
	y_scaler.inverse_transform(predictions_test),
	y_scaler.inverse_transform(y_test)
)
print(f'Test Set Error: {mae_test}')















