import numpy as np

class PhysicsEngine:
    """Physics engine for orbital mechanics calculations"""
    
    def __init__(self):
        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
        self.M_earth = 5.972e24  # Earth mass (kg)
        self.M_moon = 7.342e22   # Moon mass (kg)
        self.R_earth = 6.371e6   # Earth radius (m)
        self.R_moon = 1.737e6    # Moon radius (m)
        
        # Convert to km-based units for easier calculation
        self.G_km = self.G * 1e-9  # km³/kg⋅s²
        self.earth_moon_distance = 384400  # km (average)
        self.moon_orbital_period = 27.3 * 24 * 3600  # seconds
        
    def gravitational_force_earth(self, position, mass):
        """Calculate gravitational force from Earth"""
        # Position is in km from Earth center
        r_vec = position  # [x, y] in km
        r_magnitude = np.linalg.norm(r_vec)
        
        # Convert to meters for force calculation
        r_vec_m = r_vec * 1000  # meters
        r_magnitude_m = r_magnitude * 1000  # meters
        
        # Gravitational force magnitude
        F_magnitude = self.G * self.M_earth * mass / (r_magnitude_m**2)
        
        # Direction (toward Earth center)
        r_unit = r_vec / r_magnitude
        F_vec = -F_magnitude * r_unit
        
        return F_vec
    
    def gravitational_force_moon(self, position, mass, time):
        """Calculate gravitational force from Moon"""
        # Calculate Moon's position at given time
        moon_position = self.get_moon_position(time)
        
        # Vector from spacecraft to Moon
        r_vec = position - moon_position  # [x, y] in km
        r_magnitude = np.linalg.norm(r_vec)
        
        # Avoid division by zero and unrealistic close approaches
        if r_magnitude < 100:  # Minimum distance of 100 km
            r_magnitude = 100
            r_vec = r_vec / np.linalg.norm(r_vec) * 100
        
        # Convert to meters for force calculation
        r_vec_m = r_vec * 1000  # meters
        r_magnitude_m = r_magnitude * 1000  # meters
        
        # Gravitational force magnitude
        F_magnitude = self.G * self.M_moon * mass / (r_magnitude_m**2)
        
        # Direction (toward Moon)
        r_unit = r_vec / r_magnitude
        F_vec = -F_magnitude * r_unit
        
        return F_vec
    
    def get_moon_position(self, time):
        """Get Moon's position at given time (simplified circular orbit)"""
        # Angular velocity of Moon around Earth
        omega = 2 * np.pi / self.moon_orbital_period
        
        # Moon's position in circular orbit
        angle = omega * time
        x = self.earth_moon_distance * np.cos(angle)
        y = self.earth_moon_distance * np.sin(angle)
        
        return np.array([x, y])
    
    def total_acceleration(self, position, mass, time):
        """Calculate total acceleration at given position and time"""
        # Get forces
        F_earth = self.gravitational_force_earth(position, mass)
        F_moon = self.gravitational_force_moon(position, mass, time)
        
        # Total force
        F_total = F_earth + F_moon
        
        # Acceleration
        acceleration = F_total / mass
        
        return acceleration
    
    def orbital_velocity(self, altitude):
        """Calculate orbital velocity at given altitude above Earth"""
        # Altitude in km
        r = (self.R_earth / 1000) + altitude  # Total distance from Earth center in km
        
        # Orbital velocity in km/s
        v = np.sqrt(self.G_km * self.M_earth / r)
        
        return v
    
    def escape_velocity(self, altitude):
        """Calculate escape velocity at given altitude above Earth"""
        # Altitude in km
        r = (self.R_earth / 1000) + altitude  # Total distance from Earth center in km
        
        # Escape velocity in km/s
        v = np.sqrt(2 * self.G_km * self.M_earth / r)
        
        return v
    
    def get_mission_constants(self):
        """Get important mission constants for display"""
        return {
            'Earth Mass': f"{self.M_earth:.3e} kg",
            'Moon Mass': f"{self.M_moon:.3e} kg",
            'Earth Radius': f"{self.R_earth/1000:.1f} km",
            'Moon Radius': f"{self.R_moon/1000:.1f} km",
            'Earth-Moon Distance': f"{self.earth_moon_distance:.0f} km",
            'Gravitational Constant': f"{self.G:.3e} m³/kg⋅s²",
            'Moon Orbital Period': f"{self.moon_orbital_period/(24*3600):.1f} days"
        }
