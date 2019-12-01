import joblib as joblib

model = joblib.load('gboost_house_value_model.pkl')

house_1 = [
	2000,	# size in square feet
	3,		# number of bedrooms
	2		# number of bathrooms
]

houses = [house_1]

house_values = model.predict(houses)

predicted_value = house_values[0]

print('House details:')
print(f'- {house_1[0]} sq feet')
print(f'- {house_1[1]} bedrooms')
print(f'- {house_1[2]} bathrooms')
print(f'Estimated value: ${predicted_value:,.2f}')

