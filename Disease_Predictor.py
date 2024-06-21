import tensorflow as tf
import numpy

dataset = numpy.loadtxt('disease_prognosis_dataset.csv', delimiter=';', skiprows=1)
X = dataset[:,0:134]
y = dataset[:,134]

symptoms = [
   'abdominal_pain',
   'abnormal_menstruation',
   'acidity',
   'acute_liver_failure',
   'altered_sensorium,anxiety',
   'back_pain',
   'belly_pain',
   'blackheads',
   'bladder_discomfort',
   'blister',
   'blood_in_sputum',
   'bloody_stool',
   'blurred_and_distorted_vision',
   'breathlessness',
   'brittle_nails',
   'bruising',
   'burning_micturition',
   'chest_pain,chills',
   'cold_hands_and_feet',
   'coma,congestion',
   'constipation',
   'continuous_feel_of_urine',
   'continuous_sneezing,cough',
   'cramps',
   'dark_urine',
   'dehydration',
   'depression',
   'diarrhoea',
   'dyschromic_patches',
   'distention_of_abdomen',
   'dizziness',
   'drying_and_tingling_lips',
   'enlarged_thyroid',
   'excessive_hunger',
   'extra_marital_contacts',
   'family_history',
   'fast_heart_rate',
   'fatigue',
   'fluid_overload',
   'fluid_overload.1',
   'foul_smell_of_urine',
   'headache',
   'high_fever',
   'hip_joint_pain',
   'history_of_alcohol_consumption',
   'increased_appetite',
   'indigestion',
   'inflammatory_nails',
   'internal_itching',
   'irregular_sugar_level',
   'irritability',
   'irritation_in_anus',
   'itching',
   'joint_pain',
   'knee_pain',
   'lack_of_concentration',
   'lethargy',
   'loss_of_appetite',
   'loss_of_balance',
   'loss_of_smell',
   'loss_of_taste',
   'malaise',
   'mild_fever',
   'mood_swings',
   'movement_stiffness',
   'mucoid_sputum',
   'muscle_pain',
   'muscle_wasting',
   'muscle_weakness',
   'nausea',
   'neck_pain',
   'nodal_skin_eruptions',
   'obesity',
   'pain_behind_the_eyes',
   'pain_during_bowel_movements',
   'pain_in_anal_region',
   'painful_walking',
   'palpitations',
   'passage_of_gases',
   'patches_in_throat',
   'phlegm',
   'polyuria',
   'prominent_veins_on_calf',
   'puffy_face_and_eyes',
   'pus_filled_pimples',
   'receiving_blood_transfusion',
   'receiving_unsterile_injections',
   'red_sore_around_nose',
   'red_spots_over_body',
   'redness_of_eyes',
   'restlessness',
   'runny_nose',
   'rusty_sputum',
   'scurrying',
   'shivering',
   'silver_like_dusting',
   'sinus_pressure',
   'skin_peeling',
   'skin_rash',
   'slurred_speech',
   'small_dents_in_nails',
   'spinning_movements',
   'spotting_urination',
   'stiff_neck',
   'stomach_bleeding',
   'stomach_pain',
   'sunken_eyes',
   'sweating',
   'swelled_lymph_nodes',
   'swelling_joints',
   'swelling_of_stomach',
   'swollen_blood_vessels',
   'swollen_extremities',
   'swollen_legs',
   'throat_irritation',
   'tiredness',
   'toxic_look_(typhus)',
   'ulcers_on_tongue',
   'unsteadiness',
   'visual_disturbances',
   'vomiting',
   'watering_from_eyes',
   'weakness_in_limbs',
   'weakness_of_one_body_side',
   'weight_gain',
   'weight_loss',
   'yellow_crust_ooze',
   'yellow_urine',
   'yellowing_of_eyes',
   'yellowish_skin',
   'prognosis'
]

diseases = [
   'AIDS',
   'Acne',
   'Alcoholic hepatitis',
   'Allergy',
   'Arthritis',
   'Bronchial Asthma',
   'Cervical	spondylosis',
   'Chicken pox',
   'Chronic cholestasis',
   'Common Cold',
   'Covid',
   'Dengue',
   'Diabetes',
   'Dimorphic hemorrhoids(piles)',
   'Drug	Reaction',
   'Fungal infection',
   'GERD',
   'Gastroenteritis',
   'Heart attack',
   'Hepatitis A',
   'Hepatitis B',
   'Hepatitis C',
   'Hepatitis D',
   'Hepatitis E',
   'Hypertension',
   'Hyperthyroidism',
   'Hypoglycemia',
   'Hypothyroidism',
   'Impetigo',
   'Jaundice',
   'Malaria',
   'Migraine',
   'Osteoarthritis',
   'Paralysis',
   'Paroxysmal',
   'Peptic ulcer',
   'Pneumonia',
   'Psoriasis',
   'Tuberculosis',
   'Typhoid',
   'Urinary tract infection',
   'Varicose veins',
]

# neural network model data processing in queue
model = tf.keras.models.Sequential()
# hidden layer
# 50 neurons 81.2% accuracy
# 100 neurons 86.99% accuracy
# 500 neurons 95.41% accruracy
model.add(tf.keras.layers.Dense(12, input_dim=134, activation='relu'))
model.add(tf.keras.layers.Dense(1500, activation='relu'))
# output layer 1 neuron, you have diabetes or not
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#model.add(tf.keras.layers.Dense(len(diseases)-1, activation='sigmoid'))
# creating a neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model fitting 150 iteration, blocksize 10
model.fit(X, y, epochs=150, batch_size=10)
# model evaluation, accuracy
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# prediction based on input data
# predictions = model.predict(X)
# print(predictions)
# test_data = numpy.array([0, 85, 66, 29,0, 26, 0.351, 23])


print("Please provide the measured data to determine your disease.")
test_data=[0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0,0,0,0,0,0,0,
           0,0,0,0]  #134 zeroes for 134 symptoms

for x in range(len(symptoms)-1):
   print('(' + str(x+1) + ') ' + symptoms[x])

input1 = ''
while input1 != 0 :
   print('Enter a symptom that fits your syndrome. After you entered all your symptoms finish by entering "0"')
   input1=int(input("Symptom Nr.: "))
   test_data[input1-1] = 1

prediction = model.predict(numpy.array(test_data, ndmin=2))
#print('prediction: ' + str(prediction[0][0]))
print('You might have ' + diseases[round(prediction[0][0]*100)] + '.')