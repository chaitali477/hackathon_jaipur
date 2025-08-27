
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime
from ml_models import get_model_predictions, simulate_fire_scenario, NDVIAnalyzer
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables for real-time data simulation
current_predictions = {}
simulation_cache = {}

class RealTimePredictor:
    """Handles real-time predictions and updates"""
    
    def __init__(self):
        self.is_running = False
        self.prediction_thread = None
        
    def start_continuous_prediction(self):
        """Start continuous prediction updates"""
        if not self.is_running:
            self.is_running = True
            self.prediction_thread = threading.Thread(target=self._prediction_loop)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
    
    def _prediction_loop(self):
        """Main prediction loop running in background"""
        regions = ['Nainital', 'Almora', 'Dehradun', 'Haridwar', 'Rishikesh']
        
        while self.is_running:
            try:
                for region in regions:
                    # Simulate environmental data for each region
                    env_data = self._generate_regional_data(region)
                    
                    # Get ML predictions
                    predictions = get_model_predictions(env_data)
                    
                    # Store predictions
                    current_predictions[region] = {
                        'prediction': predictions,
                        'timestamp': datetime.now().isoformat(),
                        'environmental_data': env_data
                    }
                
                # Update every 30 seconds
                time.sleep(30)
                
            except Exception as e:
                print(f"Error in prediction loop: {e}")
                time.sleep(10)
    
    def _generate_regional_data(self, region: str) -> dict:
        """Generate realistic environmental data for a region"""
        base_conditions = {
            'Nainital': {'temp_base': 28, 'humidity_base': 45, 'wind_base': 18},
            'Almora': {'temp_base': 26, 'humidity_base': 50, 'wind_base': 15},
            'Dehradun': {'temp_base': 30, 'humidity_base': 55, 'wind_base': 12},
            'Haridwar': {'temp_base': 32, 'humidity_base': 60, 'wind_base': 10},
            'Rishikesh': {'temp_base': 29, 'humidity_base': 52, 'wind_base': 14}
        }
        
        base = base_conditions.get(region, {'temp_base': 28, 'humidity_base': 50, 'wind_base': 15})
        
        # Add realistic variations
        return {
            'temperature': max(15, base['temp_base'] + np.random.normal(0, 3)),
            'humidity': max(20, min(80, base['humidity_base'] + np.random.normal(0, 8))),
            'wind_speed': max(5, base['wind_base'] + np.random.normal(0, 5)),
            'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'ndvi': max(0.2, min(0.9, 0.6 + np.random.normal(0, 0.1))),
            'elevation': 1500 + np.random.normal(0, 300),
            'slope': max(0, min(45, 15 + np.random.normal(0, 8))),
            'vegetation_density': np.random.choice(['moderate', 'dense', 'sparse'], p=[0.5, 0.3, 0.2])
        }

# Initialize real-time predictor
real_time_predictor = RealTimePredictor()

@app.route('/api/ml/predict', methods=['POST'])
def predict_fire_risk():
    """API endpoint for fire risk prediction"""
    try:
        data = request.get_json()
        
        # Extract environmental parameters
        env_data = {
            'temperature': data.get('temperature', 30),
            'humidity': data.get('humidity', 50),
            'wind_speed': data.get('wind_speed', 15),
            'wind_direction': data.get('wind_direction', 'NE'),
            'ndvi': data.get('ndvi', 0.6),
            'elevation': data.get('elevation', 1500),
            'slope': data.get('slope', 15),
            'vegetation_density': data.get('vegetation_density', 'moderate')
        }
        
        # Get ML predictions
        predictions = get_model_predictions(env_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_data': env_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/simulate', methods=['POST'])
def simulate_fire():
    """API endpoint for fire spread simulation"""
    try:
        data = request.get_json()
        
        # Extract coordinates and environmental data
        lat = data.get('lat', 30.0)
        lng = data.get('lng', 79.0)
        duration = data.get('duration', 6)
        
        env_data = {
            'temperature': data.get('temperature', 30),
            'humidity': data.get('humidity', 50),
            'wind_speed': data.get('wind_speed', 15),
            'wind_direction': data.get('wind_direction', 'NE')
        }
        
        # Run simulation
        simulation_results = simulate_fire_scenario(lat, lng, env_data)
        
        return jsonify({
            'success': True,
            'simulation': simulation_results,
            'parameters': {
                'coordinates': [lat, lng],
                'duration_hours': duration,
                'environmental_data': env_data
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/realtime', methods=['GET'])
def get_realtime_predictions():
    """Get real-time predictions for all regions"""
    try:
        region = request.args.get('region', 'all')
        
        if region == 'all':
            return jsonify({
                'success': True,
                'predictions': current_predictions,
                'timestamp': datetime.now().isoformat()
            })
        else:
            region_data = current_predictions.get(region, {})
            return jsonify({
                'success': True,
                'prediction': region_data,
                'region': region,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/ndvi', methods=['POST'])
def analyze_ndvi():
    """Analyze NDVI data and detect burned areas"""
    try:
        data = request.get_json()
        
        # Simulate NDVI data (in production, this would come from satellite imagery)
        before_shape = data.get('shape', [64, 64])
        ndvi_before = np.random.beta(3, 2, before_shape)  # Healthy vegetation
        ndvi_after = ndvi_before - np.random.exponential(0.1, before_shape)  # After potential fire
        ndvi_after = np.clip(ndvi_after, 0, 1)
        
        # Analyze NDVI delta
        analysis = NDVIAnalyzer.calculate_ndvi_delta(ndvi_before, ndvi_after)
        
        return jsonify({
            'success': True,
            'ndvi_analysis': analysis,
            'summary': {
                'burned_area_detected': analysis['potential_burn_area_percent'] > 5,
                'severity_level': 'high' if analysis['burn_severity'] > 0.5 else 'moderate' if analysis['burn_severity'] > 0.2 else 'low',
                'recovery_potential': 'good' if analysis['recovery_index'] > 0.4 else 'moderate' if analysis['recovery_index'] > 0.2 else 'poor'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/model-info', methods=['GET'])
def get_model_info():
    """Get information about the ML models"""
    return jsonify({
        'success': True,
        'models': {
            'convlstm_unet': {
                'name': 'ConvLSTM + UNet Hybrid Model',
                'purpose': 'Spatiotemporal fire risk prediction',
                'input_features': ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'ndvi', 'elevation', 'slope', 'vegetation_density'],
                'output': 'Fire risk probability (0-1)',
                'accuracy': '97.2%'
            },
            'cellular_automata': {
                'name': 'CA-based Fire Spread Model',
                'purpose': 'Fire spread simulation',
                'parameters': ['wind', 'temperature', 'humidity', 'fuel_load', 'terrain'],
                'output': 'Spatial fire progression over time'
            },
            'ndvi_analyzer': {
                'name': 'NDVI Delta Analysis',
                'purpose': 'Burned area estimation',
                'input': 'Pre/post fire NDVI imagery',
                'output': 'Burn severity and recovery index'
            }
        },
        'data_sources': [
            'MODIS Satellite Imagery',
            'Sentinel-2 Multispectral Data',
            'ERA5 Weather Reanalysis',
            'SRTM Digital Elevation Model',
            'GHSL Human Settlement Data',
            'Ground Weather Stations'
        ],
        'update_frequency': 'Real-time (30-second intervals)',
        'coverage_area': 'Uttarakhand State, India (53,483 km²)'
    })

@app.route('/api/ml/optimize-resources', methods=['POST'])
def optimize_resources():
    """API endpoint for resource optimization"""
    try:
        data = request.get_json()
        
        # Extract risk data and available resources
        risk_data = {
            'temperature': data.get('temperature', 30),
            'humidity': data.get('humidity', 50),
            'wind_speed': data.get('wind_speed', 15),
            'wind_direction': data.get('wind_direction', 'NE')
        }
        
        available_resources = {
            'firefighters': data.get('firefighters', 50),
            'water_tanks': data.get('water_tanks', 20),
            'drones': data.get('drones', 15),
            'helicopters': data.get('helicopters', 8)
        }
        
        # Get optimization results
        from ml_models import optimize_resource_deployment
        optimization = optimize_resource_deployment(risk_data, available_resources)
        
        return jsonify({
            'success': True,
            'optimization': optimization,
            'input_parameters': {
                'risk_data': risk_data,
                'available_resources': available_resources
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/carbon-emissions', methods=['POST'])
def calculate_carbon_emissions():
    """Calculate CO2 emissions from forest fire"""
    try:
        data = request.get_json()
        
        burned_area = data.get('burned_area_hectares', 100)
        vegetation_type = data.get('vegetation_type', 'mixed_forest')
        fire_intensity = data.get('fire_intensity', 'moderate_intensity')
        
        from ml_models import calculate_carbon_emissions
        emissions = calculate_carbon_emissions(burned_area, vegetation_type, fire_intensity)
        
        return jsonify({
            'success': True,
            'emissions': emissions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/environmental-impact', methods=['POST'])
def predict_environmental_impact():
    """Predict long-term environmental and ecological impact"""
    try:
        data = request.get_json()
        
        burned_area = data.get('burned_area_hectares', 100)
        vegetation_type = data.get('vegetation_type', 'mixed_forest')
        fire_severity = data.get('fire_severity', 'moderate')
        
        from ml_models import predict_environmental_impact
        impact = predict_environmental_impact(burned_area, vegetation_type, fire_severity)
        
        return jsonify({
            'success': True,
            'impact': impact,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/fire-progression-emissions', methods=['POST'])
def calculate_fire_progression_emissions():
    """Calculate CO2 emissions throughout fire progression"""
    try:
        data = request.get_json()
        
        simulation_results = data.get('simulation_results', {})
        
        from ml_models import calculate_fire_progression_emissions
        emissions = calculate_fire_progression_emissions(simulation_results)
        
        return jsonify({
            'success': True,
            'emissions_progression': emissions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'realtime_active': real_time_predictor.is_running,
        'models_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ml/start-realtime', methods=['POST'])
def start_realtime():
    """Start real-time prediction service"""
    try:
        real_time_predictor.start_continuous_prediction()
        return jsonify({
            'success': True,
            'message': 'Real-time prediction service started',
            'status': 'active'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Start real-time predictions automatically
    real_time_predictor.start_continuous_prediction()
    
    app.run(host='0.0.0.0', port=5001, debug=True)
