from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model2_car.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        km = float(request.form['km'])
        on_road_now = float(request.form['on_road_now'])
        on_road_old = float(request.form['on_road_old'])
        condition = float(request.form['condition'])
        torque = float(request.form['torque'])
        top_speed = float(request.form['top_speed'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[km, on_road_now, on_road_old, condition, torque, top_speed]], 
                               columns=['km', 'on road now', 'on road old', 'condition', 'torque', 'top speed'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
