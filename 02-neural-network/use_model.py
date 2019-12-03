from tensorflow.keras.models import load_model
import joblib

# load the model
model = load_model('./house_value_model.h5')
X_scaler = joblib.load('./X_scaler.pkl')
y_scaler = joblib.load('./y_scaler.pkl')

house_1 = [
	2000,	# size in square feet
	3,		# number of bedrooms
	2		# number of bathrooms
]

houses = [house_1]
scaled_houses = X_scaler.transform(houses)

scaled_house_values = model.predict(scaled_houses)
unscaled_house_values = y_scaler.inverse_transform(scaled_house_values)

predicted_value = unscaled_house_values[0][0]

print('House details:')
print(f'- {house_1[0]} sq feet')
print(f'- {house_1[1]} bedrooms')
print(f'- {house_1[2]} bathrooms')
print(f'Estimated value: ${predicted_value:,.2f}')
