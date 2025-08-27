
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import json
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import requests
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EnvironmentalData:
    """Data structure for environmental parameters"""
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: str
    ndvi: float
    elevation: float
    slope: float
    vegetation_density: str
    timestamp: datetime.datetime

class DataProcessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.wind_direction_mapping = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        
        self.vegetation_mapping = {
            'sparse': 0.2, 'moderate': 0.5, 'dense': 0.8, 'very_dense': 1.0
        }
    
    def normalize_features(self, data: Dict) -> np.ndarray:
        """Normalize environmental features for ML model input"""
        # Temperature normalization (typical range: 0-50°C)
        temp_norm = min(max(data['temperature'] / 50.0, 0), 1)
        
        # Humidity normalization (0-100%)
        humidity_norm = data['humidity'] / 100.0
        
        # Wind speed normalization (typical max: 40 km/h)
        wind_norm = min(data['wind_speed'] / 40.0, 1)
        
        # Wind direction to radians
        wind_dir_degrees = self.wind_direction_mapping.get(data['wind_direction'], 0)
        wind_dir_norm = wind_dir_degrees / 360.0
        
        # NDVI is already normalized (0-1)
        ndvi_norm = data.get('ndvi', 0.5)
        
        # Elevation normalization (typical max: 4000m for Uttarakhand)
        elevation_norm = min(data.get('elevation', 1000) / 4000.0, 1)
        
        # Slope normalization (0-90 degrees)
        slope_norm = min(data.get('slope', 15) / 90.0, 1)
        
        # Vegetation density
        veg_norm = self.vegetation_mapping.get(data.get('vegetation_density', 'moderate'), 0.5)
        
        return np.array([temp_norm, humidity_norm, wind_norm, wind_dir_norm, 
                        ndvi_norm, elevation_norm, slope_norm, veg_norm])
    
    def calculate_fire_weather_index(self, temp: float, humidity: float, wind_speed: float) -> float:
        """Calculate Fire Weather Index (FWI) - simplified version"""
        # Fine Fuel Moisture Code (FFMC)
        ffmc = 85 + 0.4 * (temp - 20) - 0.5 * humidity
        ffmc = max(min(ffmc, 101), 0)
        
        # Duff Moisture Code (DMC) - simplified
        dmc = max(1, 50 - humidity * 0.5)
        
        # Drought Code (DC) - simplified
        dc = max(1, temp * 2 - humidity * 0.8)
        
        # Initial Spread Index (ISI)
        isi = 0.208 * ffmc * (0.05 + 0.1 * wind_speed)
        
        # Fire Weather Index
        fwi = 2 * np.log(isi + 1) + 0.45 * np.log(dmc + 1) + 0.15 * np.log(dc + 1)
        
        return min(max(fwi / 50.0, 0), 1)  # Normalize to 0-1

class ConvLSTMUNetModel:
    """Hybrid ConvLSTM + UNet model for fire risk prediction"""
    
    def __init__(self, input_shape=(64, 64, 8), sequence_length=5):
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the hybrid ConvLSTM + UNet architecture"""
        # Input for environmental features
        env_input = layers.Input(shape=(8,), name='environmental_features')
        
        # Input for spatial data (satellite imagery simulation)
        spatial_input = layers.Input(shape=self.input_shape, name='spatial_features')
        
        # ConvLSTM branch for temporal patterns
        convlstm = layers.ConvLSTM2D(
            filters=32, kernel_size=3, padding='same', 
            return_sequences=False, activation='tanh'
        )(tf.expand_dims(spatial_input, axis=1))
        
        # UNet-style encoder
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(spatial_input)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Bridge
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        
        # UNet decoder
        up1 = layers.UpSampling2D(size=(2, 2))(conv3)
        up1 = layers.concatenate([up1, conv2])
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
        
        up2 = layers.UpSampling2D(size=(2, 2))(conv4)
        up2 = layers.concatenate([up2, conv1])
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
        
        # Combine ConvLSTM and UNet features
        combined_spatial = layers.Add()([convlstm, conv5])
        
        # Global features from spatial data
        global_features = layers.GlobalAveragePooling2D()(combined_spatial)
        
        # Environmental features processing
        env_features = layers.Dense(32, activation='relu')(env_input)
        env_features = layers.Dense(16, activation='relu')(env_features)
        
        # Combine all features
        combined = layers.concatenate([global_features, env_features])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        
        # Output layers
        risk_output = layers.Dense(1, activation='sigmoid', name='fire_risk')(combined)
        
        # Spatial risk map output
        spatial_risk = layers.Conv2D(1, 1, activation='sigmoid', name='spatial_risk')(combined_spatial)
        
        self.model = models.Model(
            inputs=[env_input, spatial_input],
            outputs=[risk_output, spatial_risk]
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'fire_risk': 'binary_crossentropy',
                'spatial_risk': 'binary_crossentropy'
            },
            metrics={
                'fire_risk': ['accuracy'],
                'spatial_risk': ['accuracy']
            }
        )
    
    def predict_fire_risk(self, env_data: Dict, spatial_data: Optional[np.ndarray] = None) -> Dict:
        """Predict fire risk based on environmental and spatial data"""
        if spatial_data is None:
            # Generate synthetic spatial data for demo
            spatial_data = self.generate_synthetic_spatial_data(env_data)
        
        # Normalize environmental features
        processor = DataProcessor()
        env_features = processor.normalize_features(env_data).reshape(1, -1)
        spatial_features = spatial_data.reshape(1, 64, 64, 8)
        
        # Make prediction
        risk_prob, spatial_risk = self.model.predict([env_features, spatial_features], verbose=0)
        
        # Calculate additional risk metrics
        fwi = processor.calculate_fire_weather_index(
            env_data['temperature'], 
            env_data['humidity'], 
            env_data['wind_speed']
        )
        
        return {
            'overall_risk': float(risk_prob[0][0]),
            'fire_weather_index': float(fwi),
            'spatial_risk_map': spatial_risk[0].tolist(),
            'confidence': float(max(risk_prob[0][0], 1 - risk_prob[0][0])),
            'risk_category': self.categorize_risk(risk_prob[0][0])
        }
    
    def generate_synthetic_spatial_data(self, env_data: Dict) -> np.ndarray:
        """Generate synthetic spatial data for demonstration"""
        # Create a 64x64x8 array representing different data layers
        spatial_data = np.random.random((64, 64, 8))
        
        # Layer 0-2: NDVI patterns
        ndvi_base = env_data.get('ndvi', 0.5)
        spatial_data[:, :, 0] = np.random.normal(ndvi_base, 0.1, (64, 64))
        spatial_data[:, :, 1] = np.random.normal(ndvi_base - 0.1, 0.05, (64, 64))  # NDVI delta
        spatial_data[:, :, 2] = np.random.normal(ndvi_base + 0.05, 0.08, (64, 64))  # Vegetation health
        
        # Layer 3-4: Temperature and humidity patterns
        temp_norm = env_data['temperature'] / 50.0
        humidity_norm = env_data['humidity'] / 100.0
        spatial_data[:, :, 3] = np.random.normal(temp_norm, 0.1, (64, 64))
        spatial_data[:, :, 4] = np.random.normal(humidity_norm, 0.1, (64, 64))
        
        # Layer 5-6: Terrain features
        elevation_norm = 0.5  # Simplified
        slope_norm = 0.3
        spatial_data[:, :, 5] = np.random.normal(elevation_norm, 0.2, (64, 64))
        spatial_data[:, :, 6] = np.random.normal(slope_norm, 0.15, (64, 64))
        
        # Layer 7: Human activity/burned area index
        spatial_data[:, :, 7] = np.random.beta(1, 5, (64, 64))  # Low human activity
        
        return np.clip(spatial_data, 0, 1)
    
    def categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into human-readable categories"""
        if risk_score >= 0.8:
            return "very-high"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "moderate"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "very-low"

class CellularAutomataFireSpread:
    """Cellular Automata model for fire spread simulation"""
    
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.fuel_map = np.random.beta(2, 2, grid_size)  # Fuel distribution
        self.elevation_map = np.random.normal(0.5, 0.2, grid_size)
        self.moisture_map = np.random.beta(3, 2, grid_size)
        
    def ignite_fire(self, x: int, y: int):
        """Start a fire at given coordinates"""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.grid[x, y] = 1.0
    
    def spread_step(self, wind_speed: float, wind_direction: float, temperature: float, humidity: float):
        """Perform one step of fire spread simulation"""
        new_grid = self.grid.copy()
        
        # Convert wind direction to vector
        wind_x = np.cos(np.radians(wind_direction)) * wind_speed / 30.0
        wind_y = np.sin(np.radians(wind_direction)) * wind_speed / 30.0
        
        # Temperature and humidity effects
        temp_factor = min(temperature / 40.0, 1.5)
        humidity_factor = max(0.1, 1.0 - humidity / 100.0)
        
        for i in range(1, self.grid_size[0] - 1):
            for j in range(1, self.grid_size[1] - 1):
                if self.grid[i, j] > 0:  # If there's fire
                    # Check 8 neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                                if new_grid[ni, nj] == 0:  # Unburned cell
                                    # Calculate spread probability
                                    base_prob = 0.1
                                    
                                    # Fuel availability
                                    fuel_factor = self.fuel_map[ni, nj]
                                    
                                    # Slope effect (fire spreads faster uphill)
                                    slope_factor = 1.0
                                    if self.elevation_map[ni, nj] > self.elevation_map[i, j]:
                                        slope_factor = 1.5
                                    elif self.elevation_map[ni, nj] < self.elevation_map[i, j]:
                                        slope_factor = 0.7
                                    
                                    # Wind effect
                                    wind_factor = 1.0
                                    if abs(di - wind_x) < 0.5 and abs(dj - wind_y) < 0.5:
                                        wind_factor = 1.8
                                    
                                    # Moisture effect
                                    moisture_factor = max(0.1, 1.0 - self.moisture_map[ni, nj])
                                    
                                    # Calculate total probability
                                    spread_prob = (base_prob * fuel_factor * slope_factor * 
                                                 wind_factor * temp_factor * humidity_factor * moisture_factor)
                                    
                                    if np.random.random() < spread_prob:
                                        new_grid[ni, nj] = 1.0
        
        # Fire decay (burned areas become less intense over time)
        self.grid = self.grid * 0.95
        self.grid = np.maximum(self.grid, new_grid)
        
        return self.calculate_spread_metrics()
    
    def calculate_spread_metrics(self) -> Dict:
        """Calculate metrics about fire spread"""
        burned_area = np.sum(self.grid > 0.1)
        total_cells = self.grid_size[0] * self.grid_size[1]
        
        # Calculate perimeter (simplified)
        fire_cells = self.grid > 0.1
        perimeter = 0
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if fire_cells[i, j]:
                    # Check if cell is on the edge of fire
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1] 
                                and fire_cells[ni, nj]):
                                neighbors += 1
                    if neighbors < 9:  # Not completely surrounded
                        perimeter += 1
        
        return {
            'burned_area_hectares': burned_area * 0.25,  # Assuming each cell = 0.25 hectares
            'fire_perimeter_km': perimeter * 0.05,  # Rough conversion
            'fire_intensity': np.mean(self.grid[self.grid > 0]),
            'spread_rate': burned_area  # Simplified spread rate
        }

class NDVIAnalyzer:
    """NDVI Delta calculation and burned area estimation"""
    
    @staticmethod
    def calculate_ndvi_delta(ndvi_before: np.ndarray, ndvi_after: np.ndarray) -> Dict:
        """Calculate NDVI delta and estimate burned areas"""
        delta = ndvi_before - ndvi_after
        
        # Threshold for significant NDVI decrease (indicating possible burn)
        burn_threshold = 0.2
        potential_burns = delta > burn_threshold
        
        # Calculate Burned Area Index (BAI)
        bai = np.where(potential_burns, 
                      1.0 / ((0.1 - ndvi_after)**2 + (0.06 - ndvi_after)**2),
                      0)
        
        return {
            'ndvi_delta_mean': float(np.mean(delta)),
            'ndvi_delta_std': float(np.std(delta)),
            'potential_burn_area_percent': float(np.sum(potential_burns) / len(delta.flat) * 100),
            'burn_severity': float(np.mean(bai[bai > 0])) if np.any(bai > 0) else 0.0,
            'recovery_index': float(np.mean(ndvi_after[potential_burns])) if np.any(potential_burns) else 1.0
        }

class CarbonEmissionEstimator:
    """Carbon emission estimation during forest fires"""
    
    def __init__(self):
        # Carbon emission factors (kg CO2 per kg biomass burned)
        self.emission_factors = {
            'coniferous': 1.83,  # kg CO2 per kg dry biomass
            'deciduous': 1.79,
            'mixed_forest': 1.81,
            'grassland': 1.76,
            'shrubland': 1.78,
            'agricultural': 1.65
        }
        
        # Biomass density by vegetation type (kg/m²)
        self.biomass_density = {
            'coniferous': 45.0,
            'deciduous': 35.0,
            'mixed_forest': 40.0,
            'grassland': 2.5,
            'shrubland': 8.0,
            'agricultural': 3.0
        }
        
        # Combustion completeness factors
        self.combustion_completeness = {
            'low_intensity': 0.15,    # 15% of biomass consumed
            'moderate_intensity': 0.35,  # 35% consumed
            'high_intensity': 0.65,   # 65% consumed
            'extreme_intensity': 0.85  # 85% consumed
        }
    
    def estimate_co2_emissions(self, burned_area_hectares: float, vegetation_type: str, 
                              fire_intensity: str = 'moderate_intensity') -> Dict:
        """Estimate CO2 emissions from forest fire"""
        
        # Convert hectares to m²
        area_m2 = burned_area_hectares * 10000
        
        # Get parameters for vegetation type
        emission_factor = self.emission_factors.get(vegetation_type, 1.81)
        biomass_per_m2 = self.biomass_density.get(vegetation_type, 40.0)
        completeness = self.combustion_completeness.get(fire_intensity, 0.35)
        
        # Calculate total biomass
        total_biomass_kg = area_m2 * biomass_per_m2
        
        # Calculate burned biomass
        burned_biomass_kg = total_biomass_kg * completeness
        
        # Calculate CO2 emissions
        co2_emissions_kg = burned_biomass_kg * emission_factor
        co2_emissions_tonnes = co2_emissions_kg / 1000
        
        # Calculate equivalent values for context
        car_equivalent_km = co2_emissions_kg / 0.12  # Average car emits 120g CO2/km
        tree_equivalent = co2_emissions_kg / 22  # Average tree absorbs 22kg CO2/year
        
        return {
            'total_co2_emissions_kg': float(co2_emissions_kg),
            'total_co2_emissions_tonnes': float(co2_emissions_tonnes),
            'burned_area_hectares': burned_area_hectares,
            'vegetation_type': vegetation_type,
            'fire_intensity': fire_intensity,
            'biomass_burned_kg': float(burned_biomass_kg),
            'biomass_burned_tonnes': float(burned_biomass_kg / 1000),
            'equivalent_metrics': {
                'car_driving_km': float(car_equivalent_km),
                'trees_annual_absorption': float(tree_equivalent),
                'households_annual_emissions': float(co2_emissions_tonnes / 4.6)  # Average household emits 4.6 tonnes/year
            },
            'combustion_completeness_percent': float(completeness * 100)
        }
    
    def calculate_hourly_emissions(self, fire_progression: List[Dict]) -> List[Dict]:
        """Calculate hourly CO2 emission progression during fire"""
        hourly_emissions = []
        cumulative_emissions = 0
        
        for hour_data in fire_progression:
            burned_area = hour_data.get('burned_area_hectares', 0)
            vegetation = hour_data.get('vegetation_type', 'mixed_forest')
            intensity = self._determine_intensity(hour_data.get('fire_intensity', 0.5))
            
            # Calculate emissions for this hour's burned area
            if burned_area > 0:
                emissions_data = self.estimate_co2_emissions(burned_area, vegetation, intensity)
                hourly_co2 = emissions_data['total_co2_emissions_kg']
            else:
                hourly_co2 = 0
            
            cumulative_emissions += hourly_co2
            
            hourly_emissions.append({
                'hour': hour_data.get('hour', 0),
                'burned_area_hectares': burned_area,
                'hourly_co2_emissions_kg': float(hourly_co2),
                'hourly_co2_emissions_tonnes': float(hourly_co2 / 1000),
                'cumulative_co2_emissions_kg': float(cumulative_emissions),
                'cumulative_co2_emissions_tonnes': float(cumulative_emissions / 1000),
                'vegetation_type': vegetation,
                'fire_intensity': intensity
            })
        
        return hourly_emissions
    
    def _determine_intensity(self, intensity_score: float) -> str:
        """Determine fire intensity category from numerical score"""
        if intensity_score >= 0.8:
            return 'extreme_intensity'
        elif intensity_score >= 0.6:
            return 'high_intensity'
        elif intensity_score >= 0.3:
            return 'moderate_intensity'
        else:
            return 'low_intensity'

class EnvironmentalImpactPredictor:
    """Predict long-term environmental and ecological impact from forest fires"""
    
    def __init__(self):
        # Recovery time factors (years) by ecosystem type
        self.recovery_times = {
            'coniferous': {
                'vegetation_regrowth': 15,
                'full_canopy_recovery': 40,
                'soil_recovery': 25,
                'wildlife_habitat': 20,
                'carbon_sequestration': 35
            },
            'deciduous': {
                'vegetation_regrowth': 8,
                'full_canopy_recovery': 25,
                'soil_recovery': 15,
                'wildlife_habitat': 12,
                'carbon_sequestration': 20
            },
            'mixed_forest': {
                'vegetation_regrowth': 12,
                'full_canopy_recovery': 32,
                'soil_recovery': 20,
                'wildlife_habitat': 16,
                'carbon_sequestration': 28
            },
            'grassland': {
                'vegetation_regrowth': 2,
                'full_canopy_recovery': 3,
                'soil_recovery': 5,
                'wildlife_habitat': 3,
                'carbon_sequestration': 4
            },
            'shrubland': {
                'vegetation_regrowth': 5,
                'full_canopy_recovery': 8,
                'soil_recovery': 10,
                'wildlife_habitat': 6,
                'carbon_sequestration': 12
            }
        }
        
        # Severity impact multipliers
        self.severity_multipliers = {
            'low': 0.7,
            'moderate': 1.0,
            'high': 1.5,
            'severe': 2.2,
            'extreme': 3.0
        }
        
        # Ecosystem service values (USD per hectare per year)
        self.ecosystem_services = {
            'carbon_sequestration': 150,
            'water_regulation': 200,
            'air_purification': 180,
            'biodiversity_habitat': 300,
            'erosion_control': 120,
            'recreation_value': 250,
            'timber_value': 400
        }
    
    def predict_ecological_impact(self, burned_area_hectares: float, vegetation_type: str, 
                                fire_severity: str = 'moderate') -> Dict:
        """Predict comprehensive ecological impact and recovery timeline"""
        
        # Get base recovery times
        base_recovery = self.recovery_times.get(vegetation_type, self.recovery_times['mixed_forest'])
        severity_multiplier = self.severity_multipliers.get(fire_severity, 1.0)
        
        # Calculate adjusted recovery times
        recovery_timeline = {}
        for aspect, base_years in base_recovery.items():
            adjusted_years = int(base_years * severity_multiplier)
            recovery_timeline[aspect] = adjusted_years
        
        # Calculate biodiversity impact
        biodiversity_impact = self._calculate_biodiversity_impact(
            burned_area_hectares, vegetation_type, fire_severity
        )
        
        # Calculate soil impact
        soil_impact = self._calculate_soil_impact(
            burned_area_hectares, vegetation_type, fire_severity
        )
        
        # Calculate water cycle impact
        water_impact = self._calculate_water_impact(
            burned_area_hectares, vegetation_type, fire_severity
        )
        
        # Calculate economic impact
        economic_impact = self._calculate_economic_impact(
            burned_area_hectares, recovery_timeline, fire_severity
        )
        
        # Calculate carbon sequestration loss
        carbon_loss = self._calculate_carbon_sequestration_loss(
            burned_area_hectares, vegetation_type, recovery_timeline['carbon_sequestration']
        )
        
        return {
            'burned_area_hectares': burned_area_hectares,
            'vegetation_type': vegetation_type,
            'fire_severity': fire_severity,
            'recovery_timeline_years': recovery_timeline,
            'biodiversity_impact': biodiversity_impact,
            'soil_impact': soil_impact,
            'water_cycle_impact': water_impact,
            'economic_impact_usd': economic_impact,
            'carbon_sequestration_loss': carbon_loss,
            'overall_severity_score': self._calculate_overall_severity(
                biodiversity_impact, soil_impact, water_impact, economic_impact
            ),
            'recovery_phases': self._generate_recovery_phases(recovery_timeline),
            'mitigation_recommendations': self._generate_mitigation_recommendations(
                vegetation_type, fire_severity, burned_area_hectares
            )
        }
    
    def _calculate_biodiversity_impact(self, area: float, vegetation: str, severity: str) -> Dict:
        """Calculate impact on biodiversity"""
        base_species_loss = {
            'coniferous': 0.25,
            'deciduous': 0.20,
            'mixed_forest': 0.22,
            'grassland': 0.15,
            'shrubland': 0.18
        }
        
        base_loss = base_species_loss.get(vegetation, 0.22)
        severity_multiplier = self.severity_multipliers.get(severity, 1.0)
        
        species_loss_percent = min(95, base_loss * severity_multiplier * 100)
        habitat_loss_hectares = area * 0.8  # Assume 80% habitat loss
        
        return {
            'estimated_species_loss_percent': float(species_loss_percent),
            'habitat_loss_hectares': float(habitat_loss_hectares),
            'wildlife_displacement_severity': severity,
            'endangered_species_risk': 'high' if species_loss_percent > 30 else 'moderate' if species_loss_percent > 15 else 'low'
        }
    
    def _calculate_soil_impact(self, area: float, vegetation: str, severity: str) -> Dict:
        """Calculate soil degradation impact"""
        base_erosion_risk = {
            'coniferous': 0.6,
            'deciduous': 0.4,
            'mixed_forest': 0.5,
            'grassland': 0.3,
            'shrubland': 0.45
        }
        
        erosion_risk = base_erosion_risk.get(vegetation, 0.5)
        severity_multiplier = self.severity_multipliers.get(severity, 1.0)
        
        soil_loss_tonnes_per_hectare = erosion_risk * severity_multiplier * 50
        total_soil_loss = soil_loss_tonnes_per_hectare * area
        
        return {
            'soil_erosion_risk_level': 'extreme' if erosion_risk * severity_multiplier > 1.5 else 'high' if erosion_risk * severity_multiplier > 1.0 else 'moderate',
            'estimated_soil_loss_tonnes': float(total_soil_loss),
            'soil_loss_per_hectare': float(soil_loss_tonnes_per_hectare),
            'nutrient_depletion_percent': float(min(80, erosion_risk * severity_multiplier * 40)),
            'contamination_risk': 'high' if severity in ['severe', 'extreme'] else 'low'
        }
    
    def _calculate_water_impact(self, area: float, vegetation: str, severity: str) -> Dict:
        """Calculate water cycle impact"""
        base_runoff_increase = {
            'coniferous': 0.4,
            'deciduous': 0.3,
            'mixed_forest': 0.35,
            'grassland': 0.2,
            'shrubland': 0.25
        }
        
        runoff_increase = base_runoff_increase.get(vegetation, 0.35)
        severity_multiplier = self.severity_multipliers.get(severity, 1.0)
        
        return {
            'surface_runoff_increase_percent': float(runoff_increase * severity_multiplier * 100),
            'flood_risk_increase': 'high' if runoff_increase * severity_multiplier > 0.5 else 'moderate',
            'water_quality_impact': 'severe' if severity in ['severe', 'extreme'] else 'moderate',
            'downstream_sedimentation_risk': 'high' if area > 100 else 'moderate',
            'groundwater_recharge_reduction_percent': float(min(60, runoff_increase * severity_multiplier * 80))
        }
    
    def _calculate_economic_impact(self, area: float, recovery_timeline: Dict, severity: str) -> Dict:
        """Calculate economic impact of fire damage"""
        # Direct losses
        timber_loss = area * self.ecosystem_services['timber_value'] * (2 if severity in ['severe', 'extreme'] else 1)
        
        # Lost ecosystem services during recovery
        total_service_value = sum(self.ecosystem_services.values()) - self.ecosystem_services['timber_value']
        avg_recovery_years = np.mean(list(recovery_timeline.values()))
        
        ecosystem_service_loss = area * total_service_value * avg_recovery_years
        
        # Tourism and recreation loss
        recreation_loss = area * self.ecosystem_services['recreation_value'] * avg_recovery_years * 0.5
        
        total_economic_impact = timber_loss + ecosystem_service_loss + recreation_loss
        
        return {
            'direct_timber_loss_usd': float(timber_loss),
            'ecosystem_service_loss_usd': float(ecosystem_service_loss),
            'recreation_tourism_loss_usd': float(recreation_loss),
            'total_economic_impact_usd': float(total_economic_impact),
            'economic_impact_per_hectare_usd': float(total_economic_impact / area),
            'recovery_cost_estimate_usd': float(area * 2500)  # $2,500 per hectare for restoration
        }
    
    def _calculate_carbon_sequestration_loss(self, area: float, vegetation: str, recovery_years: int) -> Dict:
        """Calculate long-term carbon sequestration capacity loss"""
        annual_sequestration_rates = {  # tonnes CO2 per hectare per year
            'coniferous': 8.5,
            'deciduous': 6.2,
            'mixed_forest': 7.3,
            'grassland': 1.8,
            'shrubland': 3.2
        }
        
        annual_rate = annual_sequestration_rates.get(vegetation, 7.3)
        total_loss_tonnes = area * annual_rate * recovery_years
        
        return {
            'annual_sequestration_rate_tonnes_per_hectare': float(annual_rate),
            'total_sequestration_loss_tonnes_co2': float(total_loss_tonnes),
            'recovery_period_years': recovery_years,
            'equivalent_car_emissions_years': float(total_loss_tonnes / 4.6)  # Average car emits 4.6 tonnes CO2/year
        }
    
    def _calculate_overall_severity(self, biodiversity: Dict, soil: Dict, water: Dict, economic: Dict) -> float:
        """Calculate overall environmental impact severity score (0-10)"""
        # Weighted scoring system
        bio_score = min(10, biodiversity['estimated_species_loss_percent'] / 10)
        soil_score = min(10, 8 if soil['soil_erosion_risk_level'] == 'extreme' else 6 if soil['soil_erosion_risk_level'] == 'high' else 4)
        water_score = min(10, 8 if water['flood_risk_increase'] == 'high' else 5)
        economic_score = min(10, economic['total_economic_impact_usd'] / 1000000)  # Per million USD
        
        overall_score = (bio_score * 0.3 + soil_score * 0.25 + water_score * 0.2 + economic_score * 0.25)
        return float(min(10, overall_score))
    
    def _generate_recovery_phases(self, timeline: Dict) -> List[Dict]:
        """Generate recovery phases with milestones"""
        phases = [
            {
                'phase': 'immediate_response',
                'duration_months': 6,
                'description': 'Emergency response, erosion control, safety assessment',
                'key_activities': ['Debris removal', 'Erosion barriers', 'Safety perimeter'],
                'completion_indicators': ['Area secured', 'Immediate hazards addressed']
            },
            {
                'phase': 'early_recovery',
                'duration_months': timeline['vegetation_regrowth'] * 12,
                'description': 'Initial vegetation regrowth and soil stabilization',
                'key_activities': ['Replanting', 'Soil treatment', 'Water management'],
                'completion_indicators': ['Ground cover >50%', 'Erosion controlled']
            },
            {
                'phase': 'ecosystem_restoration',
                'duration_months': timeline['wildlife_habitat'] * 12,
                'description': 'Wildlife habitat restoration and biodiversity recovery',
                'key_activities': ['Habitat creation', 'Wildlife corridors', 'Species monitoring'],
                'completion_indicators': ['Wildlife populations returning', 'Habitat connectivity restored']
            },
            {
                'phase': 'full_recovery',
                'duration_months': timeline['full_canopy_recovery'] * 12,
                'description': 'Complete ecosystem function restoration',
                'key_activities': ['Canopy development', 'Carbon sequestration', 'Full service restoration'],
                'completion_indicators': ['Pre-fire biomass achieved', 'Full ecosystem services restored']
            }
        ]
        return phases
    
    def _generate_mitigation_recommendations(self, vegetation: str, severity: str, area: float) -> List[str]:
        """Generate specific mitigation and restoration recommendations"""
        recommendations = []
        
        # Base recommendations
        recommendations.append("Implement immediate erosion control measures using biodegradable barriers")
        recommendations.append("Conduct comprehensive soil testing for contamination and nutrient levels")
        
        if severity in ['severe', 'extreme']:
            recommendations.append("Priority reforestation with native species adapted to local conditions")
            recommendations.append("Install temporary water diversion structures to prevent flooding")
            recommendations.append("Establish wildlife corridors to facilitate species migration")
        
        if vegetation in ['coniferous', 'mixed_forest']:
            recommendations.append("Use mycorrhizal fungi inoculation to enhance root development")
            recommendations.append("Plant diverse species mix to improve resilience to future fires")
        
        if area > 100:
            recommendations.append("Implement landscape-scale restoration planning across multiple land ownerships")
            recommendations.append("Create firebreaks and fuel reduction zones during restoration")
        
        recommendations.append("Monitor water quality in downstream areas for increased sedimentation")
        recommendations.append("Establish long-term ecological monitoring plots for recovery assessment")
        
        return recommendations

class ResourceOptimizationEngine:
    """AI-powered resource optimization for firefighting deployment"""
    
    def __init__(self):
        self.resource_types = {
            'firefighters': {'capacity': 10, 'range_km': 5, 'effectiveness': 0.8},
            'water_tanks': {'capacity': 500, 'range_km': 3, 'effectiveness': 0.9},
            'drones': {'capacity': 2, 'range_km': 15, 'effectiveness': 0.6},
            'helicopters': {'capacity': 20, 'range_km': 50, 'effectiveness': 0.95}
        }
        
        self.districts = {
            'Nainital': {'lat': 29.3806, 'lng': 79.4422, 'area_km2': 4251, 'population': 954605},
            'Almora': {'lat': 29.6500, 'lng': 79.6667, 'area_km2': 3144, 'population': 622506},
            'Dehradun': {'lat': 30.3165, 'lng': 78.0322, 'area_km2': 3088, 'population': 1696694},
            'Haridwar': {'lat': 29.9457, 'lng': 78.1642, 'area_km2': 2360, 'population': 1890422},
            'Pithoragarh': {'lat': 29.5833, 'lng': 80.2167, 'area_km2': 7090, 'population': 483439},
            'Chamoli': {'lat': 30.4000, 'lng': 79.3200, 'area_km2': 8030, 'population': 391605},
            'Rudraprayag': {'lat': 30.2800, 'lng': 78.9800, 'area_km2': 1984, 'population': 242285},
            'Tehri': {'lat': 30.3900, 'lng': 78.4800, 'area_km2': 4080, 'population': 618931},
            'Pauri': {'lat': 30.1500, 'lng': 78.7800, 'area_km2': 5329, 'population': 687271},
            'Uttarkashi': {'lat': 30.7300, 'lng': 78.4500, 'area_km2': 8016, 'population': 330086},
            'Bageshwar': {'lat': 29.8400, 'lng': 79.7700, 'area_km2': 2241, 'population': 259898},
            'Champawat': {'lat': 29.3400, 'lng': 80.0900, 'area_km2': 1766, 'population': 259648},
            'Udham Singh Nagar': {'lat': 28.9750, 'lng': 79.4000, 'area_km2': 2542, 'population': 1648902}
        }
    
    def calculate_coverage_optimization(self, risk_data: Dict, available_resources: Dict) -> Dict:
        """Calculate optimal resource deployment for maximum coverage"""
        high_risk_districts = []
        moderate_risk_districts = []
        low_risk_districts = []
        
        # Categorize districts by risk level
        for district, coords in self.districts.items():
            risk_score = self._get_district_risk_score(district, risk_data)
            if risk_score >= 0.7:
                high_risk_districts.append((district, coords, risk_score))
            elif risk_score >= 0.4:
                moderate_risk_districts.append((district, coords, risk_score))
            else:
                low_risk_districts.append((district, coords, risk_score))
        
        # Sort by risk score (highest first)
        high_risk_districts.sort(key=lambda x: x[2], reverse=True)
        moderate_risk_districts.sort(key=lambda x: x[2], reverse=True)
        
        deployment_plan = self._create_deployment_plan(
            high_risk_districts, moderate_risk_districts, low_risk_districts, available_resources
        )
        
        coverage_metrics = self._calculate_coverage_metrics(deployment_plan)
        response_times = self._calculate_response_times(deployment_plan)
        
        return {
            'deployment_plan': deployment_plan,
            'coverage_metrics': coverage_metrics,
            'response_times': response_times,
            'optimization_score': self._calculate_optimization_score(coverage_metrics, response_times),
            'recommendations': self._generate_deployment_recommendations(deployment_plan, coverage_metrics)
        }
    
    def _get_district_risk_score(self, district: str, risk_data: Dict) -> float:
        """Get risk score for a specific district"""
        # Simulate risk scores based on current conditions
        base_risks = {
            'Nainital': 0.85, 'Almora': 0.68, 'Dehradun': 0.42, 'Haridwar': 0.28,
            'Pithoragarh': 0.55, 'Chamoli': 0.72, 'Rudraprayag': 0.48, 'Tehri': 0.52,
            'Pauri': 0.45, 'Uttarkashi': 0.38, 'Bageshwar': 0.41, 'Champawat': 0.36,
            'Udham Singh Nagar': 0.33
        }
        
        base_risk = base_risks.get(district, 0.5)
        
        # Apply environmental factors
        temp_factor = min(risk_data.get('temperature', 30) / 40, 1.2)
        humidity_factor = max(0.5, (100 - risk_data.get('humidity', 50)) / 100)
        wind_factor = min(risk_data.get('wind_speed', 15) / 30, 1.5)
        
        adjusted_risk = base_risk * temp_factor * humidity_factor * wind_factor
        return min(adjusted_risk, 1.0)
    
    def _create_deployment_plan(self, high_risk: List, moderate_risk: List, 
                               low_risk: List, resources: Dict) -> Dict:
        """Create optimal deployment plan for available resources"""
        deployment = {
            'firefighters': [],
            'water_tanks': [],
            'drones': [],
            'helicopters': []
        }
        
        # Priority deployment to high-risk areas
        for district, coords, risk_score in high_risk:
            # Deploy firefighters (priority resource)
            if resources.get('firefighters', 0) > 0:
                deployment['firefighters'].append({
                    'district': district,
                    'coordinates': [coords['lat'], coords['lng']],
                    'units': min(3, resources['firefighters']),
                    'risk_score': risk_score,
                    'coverage_radius': 5
                })
                resources['firefighters'] = max(0, resources['firefighters'] - 3)
            
            # Deploy water tanks
            if resources.get('water_tanks', 0) > 0:
                deployment['water_tanks'].append({
                    'district': district,
                    'coordinates': [coords['lat'], coords['lng']],
                    'units': min(2, resources['water_tanks']),
                    'risk_score': risk_score,
                    'coverage_radius': 3
                })
                resources['water_tanks'] = max(0, resources['water_tanks'] - 2)
            
            # Deploy helicopters for extreme risk areas
            if risk_score >= 0.8 and resources.get('helicopters', 0) > 0:
                deployment['helicopters'].append({
                    'district': district,
                    'coordinates': [coords['lat'], coords['lng']],
                    'units': 1,
                    'risk_score': risk_score,
                    'coverage_radius': 50
                })
                resources['helicopters'] = max(0, resources['helicopters'] - 1)
        
        # Deploy remaining resources to moderate risk areas
        for district, coords, risk_score in moderate_risk:
            if resources.get('firefighters', 0) > 0:
                deployment['firefighters'].append({
                    'district': district,
                    'coordinates': [coords['lat'], coords['lng']],
                    'units': min(2, resources['firefighters']),
                    'risk_score': risk_score,
                    'coverage_radius': 5
                })
                resources['firefighters'] = max(0, resources['firefighters'] - 2)
        
        # Deploy drones for surveillance across all areas
        drone_districts = high_risk + moderate_risk + low_risk[:3]
        for district, coords, risk_score in drone_districts:
            if resources.get('drones', 0) > 0:
                deployment['drones'].append({
                    'district': district,
                    'coordinates': [coords['lat'], coords['lng']],
                    'units': 1,
                    'risk_score': risk_score,
                    'coverage_radius': 15
                })
                resources['drones'] = max(0, resources['drones'] - 1)
        
        return deployment
    
    def _calculate_coverage_metrics(self, deployment: Dict) -> Dict:
        """Calculate coverage metrics for the deployment plan"""
        total_districts = len(self.districts)
        covered_districts = set()
        
        # Calculate coverage by resource type
        coverage_by_type = {}
        for resource_type, deployments in deployment.items():
            covered_by_resource = len(deployments)
            coverage_by_type[resource_type] = {
                'districts_covered': covered_by_resource,
                'coverage_percentage': (covered_by_resource / total_districts) * 100,
                'total_units': sum(d['units'] for d in deployments)
            }
            covered_districts.update(d['district'] for d in deployments)
        
        overall_coverage = (len(covered_districts) / total_districts) * 100
        
        return {
            'overall_coverage_percentage': overall_coverage,
            'total_districts_covered': len(covered_districts),
            'coverage_by_resource': coverage_by_type,
            'uncovered_districts': [d for d in self.districts.keys() if d not in covered_districts]
        }
    
    def _calculate_response_times(self, deployment: Dict) -> Dict:
        """Calculate estimated response times"""
        response_times = {}
        
        for resource_type, deployments in deployment.items():
            if not deployments:
                continue
                
            resource_props = self.resource_types[resource_type]
            avg_response_time = []
            
            for deploy in deployments:
                # Base response time factors
                base_time = 15  # minutes
                
                # Adjust based on resource type mobility
                if resource_type == 'helicopters':
                    response_time = base_time * 0.3  # Very fast
                elif resource_type == 'drones':
                    response_time = base_time * 0.2  # Fastest
                elif resource_type == 'firefighters':
                    response_time = base_time * 0.8  # Ground travel
                else:  # water_tanks
                    response_time = base_time * 1.2  # Slower due to equipment
                
                # Adjust based on risk score (higher risk = faster response)
                risk_factor = 2 - deploy['risk_score']  # Higher risk = lower factor = faster response
                response_time *= risk_factor
                
                avg_response_time.append(response_time)
            
            response_times[resource_type] = {
                'average_minutes': np.mean(avg_response_time) if avg_response_time else 0,
                'fastest_minutes': min(avg_response_time) if avg_response_time else 0,
                'slowest_minutes': max(avg_response_time) if avg_response_time else 0
            }
        
        # Calculate overall average
        all_times = []
        for times in response_times.values():
            if times['average_minutes'] > 0:
                all_times.append(times['average_minutes'])
        
        response_times['overall'] = {
            'average_minutes': np.mean(all_times) if all_times else 0,
            'efficiency_score': max(0, 100 - np.mean(all_times)) if all_times else 0
        }
        
        return response_times
    
    def _calculate_optimization_score(self, coverage: Dict, response: Dict) -> float:
        """Calculate overall optimization score (0-100)"""
        coverage_score = coverage['overall_coverage_percentage']
        response_score = response['overall']['efficiency_score']
        
        # Weighted average: 60% coverage, 40% response time
        optimization_score = (coverage_score * 0.6) + (response_score * 0.4)
        return round(optimization_score, 1)
    
    def _generate_deployment_recommendations(self, deployment: Dict, coverage: Dict) -> List[str]:
        """Generate actionable deployment recommendations"""
        recommendations = []
        
        # Coverage recommendations
        if coverage['overall_coverage_percentage'] < 70:
            recommendations.append("Increase resource allocation to achieve minimum 70% district coverage")
        
        if coverage['uncovered_districts']:
            uncovered = ', '.join(coverage['uncovered_districts'][:3])
            recommendations.append(f"Deploy additional resources to uncovered districts: {uncovered}")
        
        # Resource-specific recommendations
        if not deployment['helicopters']:
            recommendations.append("Consider deploying helicopters for rapid response in high-risk areas")
        
        if len(deployment['drones']) < 5:
            recommendations.append("Increase drone surveillance for better real-time monitoring")
        
        # Efficiency recommendations
        high_risk_count = sum(1 for deployments in deployment.values() 
                             for d in deployments if d['risk_score'] >= 0.7)
        if high_risk_count < 3:
            recommendations.append("Prioritize resource deployment to high-risk districts")
        
        if coverage['overall_coverage_percentage'] > 90:
            recommendations.append("Excellent coverage achieved - maintain current deployment strategy")
        
        return recommendations

class FireRiskPredictor:
    """Main class that orchestrates all ML components"""
    
    def __init__(self):
        self.convlstm_model = ConvLSTMUNetModel()
        self.ca_simulator = CellularAutomataFireSpread()
        self.data_processor = DataProcessor()
        self.resource_optimizer = ResourceOptimizationEngine()
        
        # Simulated model weights (in production, load from trained model)
        self._initialize_pretrained_weights()
    
    def _initialize_pretrained_weights(self):
        """Initialize with simulated pre-trained weights"""
        # In production, you would load actual trained weights
        # For demo, we'll use the randomly initialized weights
        pass
    
    def predict_comprehensive_risk(self, environmental_data: Dict) -> Dict:
        """Comprehensive fire risk prediction"""
        # Primary ML prediction
        ml_prediction = self.convlstm_model.predict_fire_risk(environmental_data)
        
        # Traditional fire weather index
        fwi = self.data_processor.calculate_fire_weather_index(
            environmental_data['temperature'],
            environmental_data['humidity'], 
            environmental_data['wind_speed']
        )
        
        # Combine predictions with ensemble approach
        ensemble_risk = (ml_prediction['overall_risk'] * 0.7 + fwi * 0.3)
        
        return {
            'ensemble_risk_score': float(ensemble_risk),
            'ml_prediction': ml_prediction,
            'fire_weather_index': float(fwi),
            'risk_factors': self._analyze_risk_factors(environmental_data),
            'confidence_interval': self._calculate_confidence(ml_prediction),
            'recommendations': self._generate_recommendations(ensemble_risk, environmental_data)
        }
    
    def simulate_fire_spread(self, ignition_point: Tuple[int, int], 
                           environmental_data: Dict, duration_hours: int = 6) -> Dict:
        """Simulate fire spread using cellular automata"""
        # Initialize fire
        self.ca_simulator.ignite_fire(ignition_point[0], ignition_point[1])
        
        # Convert wind direction string to degrees
        wind_dir_map = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
        wind_direction_deg = wind_dir_map.get(environmental_data['wind_direction'], 0)
        
        simulation_results = []
        
        # Run simulation for specified duration
        for hour in range(duration_hours):
            # Simulate hourly variations
            temp_variation = environmental_data['temperature'] + np.random.normal(0, 2)
            humidity_variation = max(10, environmental_data['humidity'] + np.random.normal(0, 5))
            wind_variation = max(0, environmental_data['wind_speed'] + np.random.normal(0, 3))
            
            # Perform spread step
            metrics = self.ca_simulator.spread_step(
                wind_variation, wind_direction_deg, temp_variation, humidity_variation
            )
            
            metrics['hour'] = hour
            metrics['temperature'] = temp_variation
            metrics['humidity'] = humidity_variation
            metrics['wind_speed'] = wind_variation
            
            simulation_results.append(metrics)
        
        return {
            'hourly_progression': simulation_results,
            'final_state': simulation_results[-1] if simulation_results else {},
            'fire_map': self.ca_simulator.grid.tolist()
        }
    
    def _analyze_risk_factors(self, env_data: Dict) -> Dict:
        """Analyze individual risk factors"""
        factors = {}
        
        # Temperature risk
        if env_data['temperature'] > 35:
            factors['temperature'] = 'high'
        elif env_data['temperature'] > 25:
            factors['temperature'] = 'moderate'
        else:
            factors['temperature'] = 'low'
        
        # Humidity risk (lower humidity = higher risk)
        if env_data['humidity'] < 30:
            factors['humidity'] = 'high'
        elif env_data['humidity'] < 50:
            factors['humidity'] = 'moderate'
        else:
            factors['humidity'] = 'low'
        
        # Wind risk
        if env_data['wind_speed'] > 25:
            factors['wind'] = 'high'
        elif env_data['wind_speed'] > 15:
            factors['wind'] = 'moderate'
        else:
            factors['wind'] = 'low'
        
        return factors
    
    def _calculate_confidence(self, prediction: Dict) -> Dict:
        """Calculate prediction confidence intervals"""
        base_confidence = prediction['confidence']
        
        return {
            'lower_bound': max(0, prediction['overall_risk'] - (1 - base_confidence) * 0.2),
            'upper_bound': min(1, prediction['overall_risk'] + (1 - base_confidence) * 0.2),
            'confidence_level': base_confidence
        }
    
    def _generate_recommendations(self, risk_score: float, env_data: Dict) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "Implement immediate fire prevention measures",
                "Consider evacuation planning for high-risk areas",
                "Deploy additional fire monitoring resources",
                "Issue red flag warning to public"
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "Increase fire patrol frequency",
                "Restrict outdoor burning activities",
                "Prepare firefighting resources for rapid deployment",
                "Issue fire weather watch"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Monitor weather conditions closely",
                "Maintain standard fire prevention protocols",
                "Educate public about fire safety"
            ])
        else:
            recommendations.append("Continue routine fire monitoring")
        
        # Add specific recommendations based on environmental factors
        if env_data['wind_speed'] > 20:
            recommendations.append("High wind speeds detected - extra caution with any ignition sources")
        
        if env_data['humidity'] < 30:
            recommendations.append("Very low humidity - vegetation extremely dry and flammable")
        
        return recommendations
    
    def optimize_resource_deployment(self, risk_data: Dict, available_resources: Dict) -> Dict:
        """Optimize firefighting resource deployment for maximum coverage and minimum response time"""
        return self.resource_optimizer.calculate_coverage_optimization(risk_data, available_resources)
    
    def calculate_carbon_emissions(self, burned_area_hectares: float, vegetation_type: str = 'mixed_forest', 
                                 fire_intensity: str = 'moderate_intensity') -> Dict:
        """Calculate CO2 emissions from forest fire"""
        carbon_estimator = CarbonEmissionEstimator()
        return carbon_estimator.estimate_co2_emissions(burned_area_hectares, vegetation_type, fire_intensity)
    
    def predict_environmental_impact(self, burned_area_hectares: float, vegetation_type: str = 'mixed_forest',
                                   fire_severity: str = 'moderate') -> Dict:
        """Predict long-term environmental and ecological impact"""
        impact_predictor = EnvironmentalImpactPredictor()
        return impact_predictor.predict_ecological_impact(burned_area_hectares, vegetation_type, fire_severity)
    
    def calculate_fire_progression_emissions(self, simulation_results: Dict) -> Dict:
        """Calculate CO2 emissions throughout fire progression"""
        carbon_estimator = CarbonEmissionEstimator()
        
        if 'hourly_progression' in simulation_results:
            hourly_emissions = carbon_estimator.calculate_hourly_emissions(simulation_results['hourly_progression'])
            
            total_emissions = sum(hour['cumulative_co2_emissions_tonnes'] for hour in hourly_emissions)
            max_hourly = max((hour['hourly_co2_emissions_tonnes'] for hour in hourly_emissions), default=0)
            
            return {
                'hourly_emissions': hourly_emissions,
                'total_co2_emissions_tonnes': float(total_emissions) if hourly_emissions else 0,
                'peak_hourly_emissions_tonnes': float(max_hourly),
                'emission_rate_trend': self._calculate_emission_trend(hourly_emissions)
            }
        
        return {'error': 'No hourly progression data available'}
    
    def _calculate_emission_trend(self, hourly_emissions: List[Dict]) -> str:
        """Calculate trend in emission rates"""
        if len(hourly_emissions) < 3:
            return 'insufficient_data'
        
        recent_rates = [hour['hourly_co2_emissions_tonnes'] for hour in hourly_emissions[-3:]]
        
        if recent_rates[-1] > recent_rates[0] * 1.5:
            return 'rapidly_increasing'
        elif recent_rates[-1] > recent_rates[0] * 1.1:
            return 'increasing'
        elif recent_rates[-1] < recent_rates[0] * 0.5:
            return 'rapidly_decreasing'
        elif recent_rates[-1] < recent_rates[0] * 0.9:
            return 'decreasing'
        else:
            return 'stable'

# Global model instance
fire_predictor = FireRiskPredictor()

def get_model_predictions(environmental_data: Dict) -> Dict:
    """Main function to get comprehensive fire risk predictions"""
    return fire_predictor.predict_comprehensive_risk(environmental_data)

def simulate_fire_scenario(lat: float, lng: float, env_data: Dict) -> Dict:
    """Simulate fire spread scenario at given coordinates"""
    # Convert lat/lng to grid coordinates (simplified)
    grid_x = int((lat - 29.0) * 50)  # Rough conversion for Uttarakhand region
    grid_y = int((lng - 79.0) * 50)
    
    # Ensure coordinates are within grid bounds
    grid_x = max(0, min(99, grid_x))
    grid_y = max(0, min(99, grid_y))
    
    return fire_predictor.simulate_fire_spread((grid_x, grid_y), env_data)

def optimize_resource_deployment(risk_data: Dict, available_resources: Dict) -> Dict:
    """Optimize resource deployment for maximum coverage and minimum response time"""
    return fire_predictor.optimize_resource_deployment(risk_data, available_resources)

def calculate_carbon_emissions(burned_area_hectares: float, vegetation_type: str = 'mixed_forest', 
                             fire_intensity: str = 'moderate_intensity') -> Dict:
    """Calculate CO2 emissions from forest fire"""
    return fire_predictor.calculate_carbon_emissions(burned_area_hectares, vegetation_type, fire_intensity)

def predict_environmental_impact(burned_area_hectares: float, vegetation_type: str = 'mixed_forest',
                               fire_severity: str = 'moderate') -> Dict:
    """Predict long-term environmental and ecological impact"""
    return fire_predictor.predict_environmental_impact(burned_area_hectares, vegetation_type, fire_severity)

def calculate_fire_progression_emissions(simulation_results: Dict) -> Dict:
    """Calculate CO2 emissions throughout fire progression"""
    return fire_predictor.calculate_fire_progression_emissions(simulation_results)
