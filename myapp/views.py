from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
import joblib
import pandas as pd
import numpy as np
# Load the trained model and the list of one-hot encoded columns
model = joblib.load('model.joblib')
# Define the categorical variables and their possible categories
categorical_variables = {
    'brand_name': ['apple', 'asus', 'blackview', 'blu', 'cat', 'cola', 'doogee',
       'duoqin', 'gionee', 'google', 'honor', 'huawei', 'ikall',
       'infinix', 'iqoo', 'itel', 'jio', 'lava', 'leeco', 'leitz',
       'lenovo', 'letv', 'lg', 'lyf', 'micromax', 'motorola', 'nokia',
       'nothing', 'nubia', 'oneplus', 'oppo', 'oukitel', 'poco', 'realme',
       'redmi', 'royole', 'samsung', 'sharp', 'sony', 'tcl', 'tecno',
       'tesla', 'vivo', 'xiaomi', 'zte'],
    'processor_brand': ['Not Defiend','bionic', 'dimensity', 'exynos', 'fusion',
       'google','helio', 'kirin','mediatek', 'sc9863a','snapdragon','spreadtrum','tiger','unisoc'],
    'os': ['android','ios','other']
}
def one_hot_encode(data):
    """
    One-hot encodes categorical variables in the data.

    Args:
    - data: DataFrame containing the input features.

    Returns:
    - DataFrame with categorical variables one-hot encoded.
    """
    for variable, categories in categorical_variables.items():
        # Ensure all possible categories are present in the data
        for category in categories:
            if category not in data[variable].unique():
                data[variable + '_' + category] = 0
        encoded_columns = pd.get_dummies(data[variable], prefix=variable)
        data = pd.concat([data, encoded_columns], axis=1)
        data.drop(variable, axis=1, inplace=True)
    return data

   

def predict(request):
    if request.method == 'POST':
        # Extract data from the POST request
       
        avg_rating = float(request.POST.get('avg_rating'))
        five_gig = int(request.POST.get('five_gig'))
        num_cores = float(request.POST.get('num_cores'))
        processor_speed = float(request.POST.get('processor_speed'))
        battery_capacity = float(request.POST.get('battery_capacity'))
        fast_charging_available = int(request.POST.get('fast_charging_available'))
        ram_capacity = float(request.POST.get('ram_capacity'))
        internal_memory = float(request.POST.get('internal_memory'))
        screen_size = float(request.POST.get('screen_size'))
        refresh_rate = float(request.POST.get('refresh_rate'))
        num_rear_cameras = float(request.POST.get('num_rear_cameras'))
        
        primary_camera_rear = float(request.POST.get('primary_camera_rear'))
        primary_camera_front = float(request.POST.get('primary_camera_front'))
        extended_memory_available = int(request.POST.get('extended_memory_available'))
        resolution_height = float(request.POST.get('resolution_height'))
        resolution_width = float(request.POST.get('resolution_width'))
        processor_power = float(request.POST.get('processor_power'))
        ppi = float(request.POST.get('ppi'))
        brand_name = request.POST.get('brand_name')
        processor_brand = request.POST.get('processor_brand')
        os = request.POST.get('os')
                # Check if extracted values are in defined categories
        if brand_name not in categorical_variables['brand_name']:
            return HttpResponse(f"Error: Invalid brand name '{brand_name}'")
        if processor_brand not in categorical_variables['processor_brand']:
            return HttpResponse(f"Error: Invalid processor brand '{processor_brand}'")
        if os not in categorical_variables['os']:
            return HttpResponse(f"Error: Invalid operating system '{os}'")
        # Create a DataFrame with the input features
        data = {
            
            'avg_rating': [avg_rating],
            'five_gig': [five_gig],
            
            'num_cores': [num_cores],
            'processor_speed': [processor_speed],
            'battery_capacity': [battery_capacity],
            'fast_charging_available': [fast_charging_available],
            'ram_capacity': [ram_capacity],
            'internal_memory': [internal_memory],
            'screen_size': [screen_size],
            'refresh_rate': [refresh_rate],
            'num_rear_cameras': [num_rear_cameras],
           
            'primary_camera_rear': [primary_camera_rear],
            'primary_camera_front': [primary_camera_front],
            'extended_memory_available': [extended_memory_available],
            'resolution_height': [resolution_height],
            'resolution_width': [resolution_width],
            'processor_power': [processor_power],
            'ppi': [ppi],
            'brand_name': [brand_name],
            'processor_brand': [processor_brand],
            'os': [os]
        }
        
        # Convert the dictionary to a DataFrame
        input_data = pd.DataFrame(data)
           # Apply one-hot encoding to categorical columns only
        input_data_encoded = one_hot_encode(input_data)
        training_columns=['avg_rating',
 'five_gig',
 'num_cores',
 'processor_speed',
 'battery_capacity',
 'fast_charging_available',
 'ram_capacity',
 'internal_memory',
 'screen_size',
 'refresh_rate',
 'num_rear_cameras',
 'primary_camera_rear',
 'primary_camera_front',
 'extended_memory_available',
 'resolution_height',
 'resolution_width',
 'processor_power',
 'ppi',
 'brand_name_apple',
 'brand_name_asus',
 'brand_name_blackview',
 'brand_name_blu',
 'brand_name_cat',
 'brand_name_cola',
 'brand_name_doogee',
 'brand_name_duoqin',
 'brand_name_gionee',
 'brand_name_google',
 'brand_name_honor',
 'brand_name_huawei',
 'brand_name_ikall',
 'brand_name_infinix',
 'brand_name_iqoo',
 'brand_name_itel',
 'brand_name_jio',
 'brand_name_lava',
 'brand_name_leeco',
 'brand_name_leitz',
 'brand_name_lenovo',
 'brand_name_letv',
 'brand_name_lg',
 'brand_name_lyf',
 'brand_name_micromax',
 'brand_name_motorola',
 'brand_name_nokia',
 'brand_name_nothing',
 'brand_name_nubia',
 'brand_name_oneplus',
 'brand_name_oppo',
 'brand_name_oukitel',
 'brand_name_poco',
 'brand_name_realme',
 'brand_name_redmi',
 'brand_name_royole',
 'brand_name_samsung',
 'brand_name_sharp',
 'brand_name_sony',
 'brand_name_tcl',
 'brand_name_tecno',
 'brand_name_tesla',
 'brand_name_vivo',
 'brand_name_xiaomi',
 'brand_name_zte',
 'processor_brand_Not Defiend',
 'processor_brand_bionic',
 'processor_brand_dimensity',
 'processor_brand_exynos',
 'processor_brand_fusion',
 'processor_brand_google',
 'processor_brand_helio',
 'processor_brand_kirin',
 'processor_brand_mediatek',
 'processor_brand_sc9863a',
 'processor_brand_snapdragon',
 'processor_brand_spreadtrum',
 'processor_brand_tiger',
 'processor_brand_unisoc',
 'os_android',
 'os_ios',
 'os_other']
        input_data_reordered = input_data_encoded[training_columns]

        # Make predictions
        prediction = model.predict(input_data_reordered)
        prediction=np.exp(prediction)
        prediction=0.011 *prediction
        
        # Render the prediction in the HTML template
        context = {'prediction': round(prediction[0])}
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')
