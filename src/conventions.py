# src/conventions.py
"""
Rotation Convention Documentation and Conversion Utilities
===========================================================

ROTATION CONVENTION IN THIS CODEBASE:
-------------------------------------
Our rotation follows STANDARD CARTESIAN CONVENTION:
- Positive angle = Counter-clockwise rotation from EAST (positive X-axis)
- Angle measured in degrees [0, 180°]
- Coordinate system: X = Easting, Y = Northing

VISUAL REFERENCE (Cartesian Convention):
                    N (90°)
                     ↑
                     |
        W (180°) ←---+---→ E (0°)
                     |
                     ↓
                    S (270°)

Example: 
- 0° = East direction
- 45° = Northeast direction  
- 90° = North direction
- 135° = Northwest direction

ROTATION MATRIX:
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]

Where θ is measured counter-clockwise from the positive X-axis (East).

GEOSPATIAL COORDINATE SYSTEMS:
-------------------------------
For TWD97 (Taiwan Datum 1997) and similar projected coordinate systems:
- X-axis = Easting (increases toward East)
- Y-axis = Northing (increases toward North)
- Our rotation convention directly applies

AZIMUTH VS CARTESIAN:
---------------------
Geographic Azimuth Convention (commonly used in geology/surveying):
- Measured CLOCKWISE from NORTH
- 0° = North, 90° = East, 180° = South, 270° = West

Cartesian Convention (our implementation):
- Measured COUNTER-CLOCKWISE from EAST
- 0° = East, 90° = North, 180° = West, 270° = South

CONVERSION FORMULAS:
--------------------
Cartesian → Azimuth:  azimuth = 90° - cartesian_angle
Azimuth → Cartesian:  cartesian = 90° - azimuth

(with appropriate handling of negative angles and wrapping)
"""

import numpy as np
from typing import Tuple, Dict, Any


def cartesian_to_azimuth(cartesian_angle: float) -> float:
    """
    Convert Cartesian angle to geographic azimuth.
    
    Parameters
    ----------
    cartesian_angle : float
        Angle in degrees, counter-clockwise from East [0, 360)
        
    Returns
    -------
    azimuth : float
        Angle in degrees, clockwise from North [0, 360)
        
    Examples
    --------
    >>> cartesian_to_azimuth(0)    # East
    90.0
    >>> cartesian_to_azimuth(90)   # North
    0.0
    >>> cartesian_to_azimuth(180)  # West
    270.0
    >>> cartesian_to_azimuth(45)   # Northeast
    45.0
    """
    # Convert counter-clockwise from East to clockwise from North
    azimuth = 90.0 - cartesian_angle
    
    # Normalize to [0, 360)
    azimuth = azimuth % 360.0
    
    return azimuth


def azimuth_to_cartesian(azimuth: float) -> float:
    """
    Convert geographic azimuth to Cartesian angle.
    
    Parameters
    ----------
    azimuth : float
        Angle in degrees, clockwise from North [0, 360)
        
    Returns
    -------
    cartesian_angle : float
        Angle in degrees, counter-clockwise from East [0, 360)
        
    Examples
    --------
    >>> azimuth_to_cartesian(0)    # North
    90.0
    >>> azimuth_to_cartesian(90)   # East
    0.0
    >>> azimuth_to_cartesian(180)  # South
    270.0
    >>> azimuth_to_cartesian(45)   # Northeast
    45.0
    """
    # Convert clockwise from North to counter-clockwise from East
    cartesian = 90.0 - azimuth
    
    # Normalize to [0, 360)
    cartesian = cartesian % 360.0
    
    return cartesian


def format_rotation_summary(cartesian_angle: float) -> Dict[str, Any]:
    """
    Generate comprehensive rotation angle summary in both conventions.
    
    Parameters
    ----------
    cartesian_angle : float
        Rotation angle from GPR model (counter-clockwise from East)
        
    Returns
    -------
    summary : dict
        Dictionary with angles in both conventions and directional description
    """
    # Normalize to [0, 360)
    cartesian_normalized = cartesian_angle % 360.0
    
    # Convert to azimuth
    azimuth = cartesian_to_azimuth(cartesian_normalized)
    
    # Determine cardinal direction
    def get_direction(angle: float, convention: str = "azimuth") -> str:
        """Get compass direction from angle."""
        if convention == "azimuth":
            # Azimuth: 0° = N, 90° = E, 180° = S, 270° = W
            if angle < 22.5 or angle >= 337.5:
                return "N"
            elif 22.5 <= angle < 67.5:
                return "NE"
            elif 67.5 <= angle < 112.5:
                return "E"
            elif 112.5 <= angle < 157.5:
                return "SE"
            elif 157.5 <= angle < 202.5:
                return "S"
            elif 202.5 <= angle < 247.5:
                return "SW"
            elif 247.5 <= angle < 292.5:
                return "W"
            else:
                return "NW"
        else:  # cartesian
            # Cartesian: 0° = E, 90° = N, 180° = W, 270° = S
            if angle < 22.5 or angle >= 337.5:
                return "E"
            elif 22.5 <= angle < 67.5:
                return "NE"
            elif 67.5 <= angle < 112.5:
                return "N"
            elif 112.5 <= angle < 157.5:
                return "NW"
            elif 157.5 <= angle < 202.5:
                return "W"
            elif 202.5 <= angle < 247.5:
                return "SW"
            elif 247.5 <= angle < 292.5:
                return "S"
            else:
                return "SE"
    
    direction_cartesian = get_direction(cartesian_normalized, "cartesian")
    direction_azimuth = get_direction(azimuth, "azimuth")
    
    return {
        "cartesian_angle_deg": round(cartesian_normalized, 2),
        "azimuth_deg": round(azimuth, 2),
        "primary_direction_cartesian": direction_cartesian,
        "primary_direction_azimuth": direction_azimuth,
        "convention": "Cartesian (counter-clockwise from East)",
        "interpretation": f"Primary correlation axis trending {direction_azimuth} (azimuth {azimuth:.1f}°)"
    }
