import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TrajectoryVisualizer:
    """Visualize spacecraft trajectory and mission data"""
    
    def __init__(self):
        self.earth_radius = 6371  # km
        self.moon_radius = 1737   # km
        self.earth_moon_distance = 384400  # km
    
    def plot_trajectory(self, trajectory_data):
        """Create interactive trajectory plot"""
        positions = trajectory_data['positions']
        times = trajectory_data['times']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trajectory Overview', 'Altitude vs Time', 'Speed vs Time', 'Energy vs Time'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # Main trajectory plot
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='lines',
                name='Spacecraft Trajectory',
                line=dict(color='blue', width=2),
                hovertemplate='X: %{x:.0f} km<br>Y: %{y:.0f} km<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Earth
        earth_circle = self.create_circle(0, 0, self.earth_radius, 'Earth', 'green')
        fig.add_trace(earth_circle, row=1, col=1)
        
        # Add Moon positions
        moon_positions = np.array([self.get_moon_position(t) for t in times])
        fig.add_trace(
            go.Scatter(
                x=moon_positions[:, 0],
                y=moon_positions[:, 1],
                mode='lines',
                name='Moon Orbit',
                line=dict(color='gray', width=1, dash='dash'),
                hovertemplate='Moon X: %{x:.0f} km<br>Moon Y: %{y:.0f} km<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Moon at final position
        final_moon_pos = moon_positions[-1]
        moon_circle = self.create_circle(final_moon_pos[0], final_moon_pos[1], self.moon_radius, 'Moon', 'lightgray')
        fig.add_trace(moon_circle, row=1, col=1)
        
        # Altitude vs Time
        distances_from_earth = np.linalg.norm(positions, axis=1)
        altitudes = distances_from_earth - self.earth_radius
        
        fig.add_trace(
            go.Scatter(
                x=times / 3600,  # Convert to hours
                y=altitudes,
                mode='lines',
                name='Altitude',
                line=dict(color='red'),
                hovertemplate='Time: %{x:.1f} h<br>Altitude: %{y:.0f} km<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Speed vs Time
        speeds = np.linalg.norm(trajectory_data['velocities'], axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=times / 3600,  # Convert to hours
                y=speeds,
                mode='lines',
                name='Speed',
                line=dict(color='orange'),
                hovertemplate='Time: %{x:.1f} h<br>Speed: %{y:.3f} km/s<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Energy vs Time (simplified)
        kinetic_energy = 0.5 * trajectory_data['mass'] * speeds**2
        potential_energy = -6.67430e-11 * 5.972e24 * trajectory_data['mass'] / (distances_from_earth * 1000)
        total_energy = kinetic_energy + potential_energy
        
        fig.add_trace(
            go.Scatter(
                x=times / 3600,
                y=total_energy / 1e9,  # Convert to GJ
                mode='lines',
                name='Total Energy',
                line=dict(color='purple'),
                hovertemplate='Time: %{x:.1f} h<br>Energy: %{y:.1f} GJ<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Moon Launch Trajectory Analysis',
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes
        fig.update_xaxes(title_text="X Position (km)", row=1, col=1)
        fig.update_yaxes(title_text="Y Position (km)", row=1, col=1)
        fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Altitude (km)", row=1, col=2)
        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="Speed (km/s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
        fig.update_yaxes(title_text="Energy (GJ)", row=2, col=2)
        
        # Equal aspect ratio for trajectory plot
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        
        return fig
    
    def create_circle(self, x_center, y_center, radius, name, color):
        """Create a circle for celestial bodies"""
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = x_center + radius * np.cos(theta)
        y_circle = y_center + radius * np.sin(theta)
        
        return go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=f'{name}<br>Center: ({x_center:.0f}, {y_center:.0f})<br>Radius: {radius:.0f} km<extra></extra>'
        )
    
    def get_moon_position(self, time):
        """Get Moon position at given time"""
        moon_orbital_period = 27.3 * 24 * 3600  # seconds
        omega = 2 * np.pi / moon_orbital_period
        angle = omega * time
        x = self.earth_moon_distance * np.cos(angle)
        y = self.earth_moon_distance * np.sin(angle)
        return np.array([x, y])
    
    def plot_forces_diagram(self, trajectory_data, step_index):
        """Plot forces diagram for a specific step"""
        position = trajectory_data['positions'][step_index]
        time = trajectory_data['times'][step_index]
        
        # Calculate forces (simplified visualization)
        # Earth force direction
        earth_force_dir = -position / np.linalg.norm(position)
        
        # Moon force direction
        moon_pos = self.get_moon_position(time)
        moon_vec = position - moon_pos
        moon_force_dir = -moon_vec / np.linalg.norm(moon_vec)
        
        # Create force diagram
        fig = go.Figure()
        
        # Spacecraft position
        fig.add_trace(go.Scatter(
            x=[position[0]],
            y=[position[1]],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Spacecraft'
        ))
        
        # Earth force arrow
        earth_arrow_end = position + earth_force_dir * 10000  # Scale for visibility
        fig.add_trace(go.Scatter(
            x=[position[0], earth_arrow_end[0]],
            y=[position[1], earth_arrow_end[1]],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            name='Earth Force'
        ))
        
        # Moon force arrow
        moon_arrow_end = position + moon_force_dir * 5000  # Scale for visibility
        fig.add_trace(go.Scatter(
            x=[position[0], moon_arrow_end[0]],
            y=[position[1], moon_arrow_end[1]],
            mode='lines+markers',
            line=dict(color='gray', width=3),
            marker=dict(size=8),
            name='Moon Force'
        ))
        
        # Add Earth and Moon
        earth_circle = self.create_circle(0, 0, self.earth_radius, 'Earth', 'green')
        fig.add_trace(earth_circle)
        
        moon_circle = self.create_circle(moon_pos[0], moon_pos[1], self.moon_radius, 'Moon', 'lightgray')
        fig.add_trace(moon_circle)
        
        fig.update_layout(
            title=f'Force Diagram at Step {step_index + 1}',
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True
        )
        
        return fig
    
    def plot_mission_phases(self, trajectory_data):
        """Plot trajectory with mission phases highlighted"""
        positions = trajectory_data['positions']
        times = trajectory_data['times']
        distances = np.linalg.norm(positions, axis=1)
        
        # Define phase boundaries
        phases = []
        current_phase = 0
        phase_names = ['Liftoff', 'Earth Orbit', 'Trans-lunar', 'Lunar Approach']
        phase_colors = ['red', 'blue', 'orange', 'purple']
        
        # Simple phase detection based on distance
        for i in range(len(distances)):
            if distances[i] < 10000:  # Within 10,000 km of Earth
                phase = 0 if i < 100 else 1
            elif distances[i] < 100000:  # Trans-lunar injection
                phase = 2
            else:  # Lunar approach
                phase = 3
            
            phases.append(phase)
        
        fig = go.Figure()
        
        # Plot trajectory segments by phase
        phases = np.array(phases)
        for phase_idx in range(4):
            phase_mask = phases == phase_idx
            if np.any(phase_mask):
                phase_positions = positions[phase_mask]
                fig.add_trace(go.Scatter(
                    x=phase_positions[:, 0],
                    y=phase_positions[:, 1],
                    mode='lines',
                    name=phase_names[phase_idx],
                    line=dict(color=phase_colors[phase_idx], width=3)
                ))
        
        # Add celestial bodies
        earth_circle = self.create_circle(0, 0, self.earth_radius, 'Earth', 'green')
        fig.add_trace(earth_circle)
        
        # Moon orbit
        moon_positions = np.array([self.get_moon_position(t) for t in times])
        fig.add_trace(go.Scatter(
            x=moon_positions[:, 0],
            y=moon_positions[:, 1],
            mode='lines',
            name='Moon Orbit',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Mission Phases',
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True
        )
        
        return fig
