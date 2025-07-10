import numpy as np
from physics_engine import PhysicsEngine

class TrajectoryCalculator:
    """Calculate spacecraft trajectory using Euler's method"""
    
    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.spacecraft_mass = 50000  # kg (typical spacecraft mass)
    
    def calculate_trajectory(self, initial_position, initial_velocity, step_size, max_time):
        """
        Calculate trajectory using Euler's method
        
        Args:
            initial_position: [x, y] in km from Earth center
            initial_velocity: [vx, vy] in km/s
            step_size: time step in seconds
            max_time: maximum simulation time in seconds
        
        Returns:
            Dictionary with trajectory data
        """
        # Initialize arrays
        num_steps = int(max_time / step_size) + 1
        times = np.zeros(num_steps)
        positions = np.zeros((num_steps, 2))
        velocities = np.zeros((num_steps, 2))
        accelerations = np.zeros((num_steps, 2))
        
        # Set initial conditions
        positions[0] = initial_position
        velocities[0] = initial_velocity
        accelerations[0] = self.physics.total_acceleration(initial_position, self.spacecraft_mass, 0)
        
        # Euler's method integration
        for i in range(num_steps - 1):
            current_time = i * step_size
            current_position = positions[i]
            current_velocity = velocities[i]
            
            # Calculate acceleration at current state
            current_acceleration = self.physics.total_acceleration(
                current_position, self.spacecraft_mass, current_time
            )
            accelerations[i] = current_acceleration
            
            # Euler's method update
            # r_(n+1) = r_n + v_n * dt
            positions[i + 1] = current_position + current_velocity * step_size
            
            # v_(n+1) = v_n + a_n * dt
            velocities[i + 1] = current_velocity + current_acceleration * step_size
            
            # Update time
            times[i + 1] = (i + 1) * step_size
            
            # Check for collision or extreme distance
            distance_from_earth = np.linalg.norm(positions[i + 1])
            if distance_from_earth < 6371:  # Below Earth surface
                print(f"Collision with Earth at step {i + 1}")
                break
            elif distance_from_earth > 1000000:  # Too far from Earth-Moon system
                print(f"Trajectory extends beyond reasonable bounds at step {i + 1}")
                break
        
        # Calculate final acceleration
        if times[-1] > 0:
            accelerations[-1] = self.physics.total_acceleration(
                positions[-1], self.spacecraft_mass, times[-1]
            )
        
        # Trim arrays to actual length
        actual_length = np.where(times > 0)[0][-1] + 1 if np.any(times > 0) else 1
        
        return {
            'times': times[:actual_length],
            'positions': positions[:actual_length],
            'velocities': velocities[:actual_length],
            'accelerations': accelerations[:actual_length],
            'step_size': step_size,
            'mass': self.spacecraft_mass,
            'initial_position': initial_position,
            'initial_velocity': initial_velocity
        }
    
    def analyze_trajectory(self, trajectory_data):
        """Analyze trajectory for key mission parameters"""
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        times = trajectory_data['times']
        
        # Calculate distances and speeds
        distances_from_earth = np.linalg.norm(positions, axis=1)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Find key points
        max_altitude_idx = np.argmax(distances_from_earth)
        max_speed_idx = np.argmax(speeds)
        
        # Calculate energy
        kinetic_energy = 0.5 * trajectory_data['mass'] * speeds**2
        potential_energy_earth = -self.physics.G_km * self.physics.M_earth * trajectory_data['mass'] / distances_from_earth
        
        # Moon positions for potential energy calculation
        moon_positions = np.array([self.physics.get_moon_position(t) for t in times])
        distances_from_moon = np.linalg.norm(positions - moon_positions, axis=1)
        potential_energy_moon = -self.physics.G_km * self.physics.M_moon * trajectory_data['mass'] / distances_from_moon
        
        total_energy = kinetic_energy + potential_energy_earth + potential_energy_moon
        
        analysis = {
            'max_altitude': distances_from_earth[max_altitude_idx] - 6371,  # km above Earth surface
            'max_altitude_time': times[max_altitude_idx] / 3600,  # hours
            'max_speed': speeds[max_speed_idx],  # km/s
            'max_speed_time': times[max_speed_idx] / 3600,  # hours
            'final_distance': distances_from_earth[-1],  # km from Earth center
            'final_speed': speeds[-1],  # km/s
            'flight_time': times[-1] / 3600,  # hours
            'total_energy': total_energy,
            'distances_from_earth': distances_from_earth,
            'distances_from_moon': distances_from_moon,
            'speeds': speeds,
            'moon_positions': moon_positions
        }
        
        return analysis
    
    def get_trajectory_at_time(self, trajectory_data, target_time):
        """Get trajectory state at specific time"""
        times = trajectory_data['times']
        
        # Find closest time index
        closest_idx = np.argmin(np.abs(times - target_time))
        
        return {
            'index': closest_idx,
            'time': times[closest_idx],
            'position': trajectory_data['positions'][closest_idx],
            'velocity': trajectory_data['velocities'][closest_idx],
            'acceleration': trajectory_data['accelerations'][closest_idx]
        }
    
    def predict_landing_conditions(self, trajectory_data):
        """Predict if and when spacecraft might reach Moon vicinity"""
        positions = trajectory_data['positions']
        times = trajectory_data['times']
        
        # Calculate Moon positions
        moon_positions = np.array([self.physics.get_moon_position(t) for t in times])
        
        # Calculate distances to Moon
        distances_to_moon = np.linalg.norm(positions - moon_positions, axis=1)
        
        # Find closest approach to Moon
        closest_approach_idx = np.argmin(distances_to_moon)
        closest_distance = distances_to_moon[closest_approach_idx]
        
        # Check if trajectory reaches Moon vicinity (within 10,000 km)
        moon_vicinity = closest_distance < 10000
        
        return {
            'reaches_moon_vicinity': moon_vicinity,
            'closest_approach_distance': closest_distance,
            'closest_approach_time': times[closest_approach_idx] / 3600,  # hours
            'closest_approach_position': positions[closest_approach_idx],
            'moon_position_at_closest': moon_positions[closest_approach_idx]
        }
