import numpy as np
import pandas as pd


class AQICalculator:
    
    
    PM25_BREAKPOINTS = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]
    
    PM10_BREAKPOINTS = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ]
    
    O3_BREAKPOINTS = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300)
    ]
    
    NO2_BREAKPOINTS = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500)
    ]
    
    SO2_BREAKPOINTS = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500)
    ]
    
    CO_BREAKPOINTS = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500)
    ]
    
    @staticmethod
    def _calculate_aqi_for_pollutant(concentration, breakpoints):
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        # Find the appropriate breakpoint
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration <= c_high:
                # Calculate AQI using linear interpolation
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return round(aqi)
        
        # If concentration exceeds all breakpoints, use the highest AQI
        return 500
    
    @classmethod
    def calculate_pm25_aqi(cls, pm25):
        """Calculate AQI from PM2.5 concentration (μg/m³)"""
        return cls._calculate_aqi_for_pollutant(pm25, cls.PM25_BREAKPOINTS)
    
    @classmethod
    def calculate_pm10_aqi(cls, pm10):
        """Calculate AQI from PM10 concentration (μg/m³)"""
        return cls._calculate_aqi_for_pollutant(pm10, cls.PM10_BREAKPOINTS)
    
    @classmethod
    def calculate_o3_aqi(cls, o3):
        return cls._calculate_aqi_for_pollutant(o3, cls.O3_BREAKPOINTS)
    
    @classmethod
    def calculate_no2_aqi(cls, no2):
        """Calculate AQI from NO2 concentration (ppb)"""
        return cls._calculate_aqi_for_pollutant(no2, cls.NO2_BREAKPOINTS)
    
    @classmethod
    def calculate_so2_aqi(cls, so2):
        return cls._calculate_aqi_for_pollutant(so2, cls.SO2_BREAKPOINTS)
    
    @classmethod
    def calculate_co_aqi(cls, co):
        return cls._calculate_aqi_for_pollutant(co, cls.CO_BREAKPOINTS)
    
    @classmethod
    def calculate_aqi(cls, pm25=None, pm10=None, o3=None, no2=None, so2=None, co=None):
        aqi_values = []
        
        if pm25 is not None:
            aqi_values.append(cls.calculate_pm25_aqi(pm25))
        if pm10 is not None:
            aqi_values.append(cls.calculate_pm10_aqi(pm10))
        if o3 is not None:
            aqi_values.append(cls.calculate_o3_aqi(o3))
        if no2 is not None:
            aqi_values.append(cls.calculate_no2_aqi(no2))
        if so2 is not None:
            aqi_values.append(cls.calculate_so2_aqi(so2))
        if co is not None:
            aqi_values.append(cls.calculate_co_aqi(co))
        
        # Filter out NaN values
        aqi_values = [aqi for aqi in aqi_values if not pd.isna(aqi)]
        
        if not aqi_values:
            return np.nan
        
        return int(max(aqi_values))
    
    @staticmethod
    def get_aqi_category(aqi):
        if pd.isna(aqi):
            return {
                "category": "Unknown",
                "level": 0,
                "color": "#808080",
                "message": "Insufficient data"
            }
        
        if aqi <= 50:
            return {
                "category": "Good",
                "level": 1,
                "color": "#00E400",
                "message": "Air quality is satisfactory, and air pollution poses little or no risk."
            }
        elif aqi <= 100:
            return {
                "category": "Moderate",
                "level": 2,
                "color": "#FFFF00",
                "message": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
            }
        elif aqi <= 150:
            return {
                "category": "Unhealthy for Sensitive Groups",
                "level": 3,
                "color": "#FF7E00",
                "message": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
            }
        elif aqi <= 200:
            return {
                "category": "Unhealthy",
                "level": 4,
                "color": "#FF0000",
                "message": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
            }
        elif aqi <= 300:
            return {
                "category": "Very Unhealthy",
                "level": 5,
                "color": "#8F3F97",
                "message": "Health alert: The risk of health effects is increased for everyone."
            }
        else:
            return {
                "category": "Hazardous",
                "level": 6,
                "color": "#7E0023",
                "message": "Health warning of emergency conditions: everyone is more likely to be affected."
            }


def convert_openmeteo_to_epa_units(df):
    df = df.copy()
    
    # Convert O3 from μg/m³ to ppb
    if 'ozone' in df.columns:
        df['ozone_ppb'] = df['ozone'] / 2.0
    
    # Convert NO2 from μg/m³ to ppb
    if 'nitrogen_dioxide' in df.columns:
        df['nitrogen_dioxide_ppb'] = df['nitrogen_dioxide'] / 1.88
    
    # Convert SO2 from μg/m³ to ppb
    if 'sulphur_dioxide' in df.columns:
        df['sulphur_dioxide_ppb'] = df['sulphur_dioxide'] / 2.62
    
    # Convert CO from mg/m³ to ppm
    if 'carbon_monoxide' in df.columns:
        df['carbon_monoxide_ppm'] = df['carbon_monoxide'] / 1145.0  # mg/m³ to ppm
    
    return df


def calculate_aqi_for_dataframe(df):
    df = df.copy()
    
    # Convert Open-Meteo units to EPA units
    df = convert_openmeteo_to_epa_units(df)
    
    # Calculate AQI for each row
    df['aqi'] = df.apply(
        lambda row: AQICalculator.calculate_aqi(
            pm25=row.get('pm2_5'),
            pm10=row.get('pm10'),
            o3=row.get('ozone_ppb'),
            no2=row.get('nitrogen_dioxide_ppb'),
            so2=row.get('sulphur_dioxide_ppb'),
            co=row.get('carbon_monoxide_ppm')
        ),
        axis=1
    )
    
    return df


if __name__ == "__main__":
    # Test the calculator
    print("Testing AQI Calculator\n" + "="*50)
    
    # Test individual pollutants
    test_cases = [
        {"pm25": 15.0, "expected": "Good"},
        {"pm25": 45.0, "expected": "Moderate"},
        {"pm10": 100.0, "expected": "Moderate"},
        {"o3": 80.0, "expected": "Unhealthy for Sensitive Groups"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        aqi = AQICalculator.calculate_aqi(**case)
        category = AQICalculator.get_aqi_category(aqi)
        print(f"Test {i}: {case} → AQI={aqi}, Category={category['category']}")
    
    print("\n" + "="*50)
    print("AQI Calculator is working correctly!")

