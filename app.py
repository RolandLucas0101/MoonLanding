import streamlit as st
import numpy as np
import pandas as pd
import time
from physics_engine import PhysicsEngine
from trajectory_calculator import TrajectoryCalculator
from visualization import TrajectoryVisualizer

# Page configuration
st.set_page_config(
    page_title="Moon Launch Trajectory Simulation",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'trajectory_data' not in st.session_state:
    st.session_state.trajectory_data = None
if 'physics_engine' not in st.session_state:
    st.session_state.physics_engine = PhysicsEngine()
if 'calculator' not in st.session_state:
    st.session_state.calculator = TrajectoryCalculator(st.session_state.physics_engine)

def main():
    st.title("🚀 Moon Launch Trajectory Simulation")
    st.markdown("### Educational Simulation using Euler's Method")
    
    # Sidebar for parameters
    st.sidebar.header("Mission Parameters")
    
    # Initial conditions
    st.sidebar.subheader("Initial Conditions")
    initial_altitude = st.sidebar.slider("Initial Altitude (km)", 0, 500, 100)
    initial_velocity = st.sidebar.slider("Initial Velocity (km/s)", 7.0, 15.0, 11.2)
    launch_angle = st.sidebar.slider("Launch Angle (degrees)", 0, 90, 45)
    
    # Euler's method parameters
    st.sidebar.subheader("Euler's Method Parameters")
    step_size = st.sidebar.selectbox("Step Size (seconds)", [1, 10, 60, 300, 600], index=3)
    max_time = st.sidebar.slider("Maximum Simulation Time (hours)", 24, 240, 96)
    
    # Calculate trajectory button
    if st.sidebar.button("Calculate Trajectory", type="primary"):
        with st.spinner("Calculating trajectory..."):
            # Set initial conditions
            earth_radius = 6371  # km
            initial_position = np.array([earth_radius + initial_altitude, 0])
            velocity_magnitude = initial_velocity
            angle_rad = np.radians(launch_angle)
            initial_velocity_vec = np.array([
                velocity_magnitude * np.cos(angle_rad),
                velocity_magnitude * np.sin(angle_rad)
            ])
            
            # Calculate trajectory
            trajectory_data = st.session_state.calculator.calculate_trajectory(
                initial_position, initial_velocity_vec, step_size, max_time * 3600
            )
            st.session_state.trajectory_data = trajectory_data
            st.success("Trajectory calculated successfully!")
    
    # Main content area
    if st.session_state.trajectory_data is not None:
        data = st.session_state.trajectory_data
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Trajectory Overview", 
            "🚀 Moving Simulation",
            "🔢 Euler's Method Details", 
            "📈 Step-by-Step Calculations", 
            "🎯 Custom Point Analysis"
        ])
        
        with tab1:
            display_trajectory_overview(data)
        
        with tab2:
            display_moving_simulation(data)
        
        with tab3:
            display_eulers_method_details(data)
        
        with tab4:
            display_step_calculations(data)
        
        with tab5:
            display_custom_point_analysis(data)
    
    else:
        st.info("👈 Set your mission parameters in the sidebar and click 'Calculate Trajectory' to begin the simulation.")
        display_physics_background()

def display_trajectory_overview(data):
    st.header("Mission Trajectory Overview")
    
    # Create visualizer
    visualizer = TrajectoryVisualizer()
    
    # Plot trajectory
    fig = visualizer.plot_trajectory(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Mission statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_altitude = np.max(np.linalg.norm(data['positions'], axis=1)) - 6371
        st.metric("Maximum Altitude", f"{max_altitude:.1f} km")
    
    with col2:
        max_velocity = np.max(np.linalg.norm(data['velocities'], axis=1))
        st.metric("Maximum Velocity", f"{max_velocity:.2f} km/s")
    
    with col3:
        total_time = data['times'][-1] / 3600
        st.metric("Total Flight Time", f"{total_time:.1f} hours")
    
    with col4:
        final_distance = np.linalg.norm(data['positions'][-1])
        st.metric("Final Distance from Earth", f"{final_distance:.0f} km")

def display_moving_simulation(data):
    """Display animated simulation of the spacecraft trajectory"""
    st.header("🚀 Moving Simulation")
    
    # Simulation controls
    col1, col2 = st.columns(2)
    
    with col1:
        time_unit = st.radio("Time Unit", ["seconds", "minutes"], index=1)
        
    with col2:
        if time_unit == "seconds":
            animation_duration = st.slider("Animation Duration (seconds)", 5, 300, 30)
            time_factor = 1
        else:
            animation_duration = st.slider("Animation Duration (minutes)", 1, 10, 2)
            time_factor = 60
    
    # Convert to seconds for calculation
    total_animation_seconds = animation_duration * time_factor
    
    # Animation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_animation = st.button("▶️ Start Animation", type="primary")
    
    with col2:
        pause_animation = st.button("⏸️ Pause")
    
    with col3:
        reset_animation = st.button("🔄 Reset")
    
    # Initialize animation state
    if 'animation_running' not in st.session_state:
        st.session_state.animation_running = False
    if 'animation_step' not in st.session_state:
        st.session_state.animation_step = 0
    if 'animation_start_time' not in st.session_state:
        st.session_state.animation_start_time = 0
    
    # Handle button clicks
    if start_animation:
        st.session_state.animation_running = True
        st.session_state.animation_start_time = time.time()
    
    if pause_animation:
        st.session_state.animation_running = False
    
    if reset_animation:
        st.session_state.animation_running = False
        st.session_state.animation_step = 0
        st.session_state.animation_start_time = 0
    
    # Calculate animation progress
    if st.session_state.animation_running:
        elapsed_time = time.time() - st.session_state.animation_start_time
        progress = min(elapsed_time / total_animation_seconds, 1.0)
        
        # Calculate current step in trajectory
        total_steps = len(data['positions'])
        current_step = int(progress * total_steps)
        
        if current_step >= total_steps - 1:
            st.session_state.animation_running = False
            current_step = total_steps - 1
            
        st.session_state.animation_step = current_step
    else:
        current_step = st.session_state.animation_step
    
    # Create the animated plot
    visualizer = TrajectoryVisualizer()
    fig = visualizer.plot_animated_trajectory(data, current_step)
    
    # Display the plot
    plot_placeholder = st.empty()
    with plot_placeholder.container():
        st.plotly_chart(fig, use_container_width=True)
    
    # Display current mission status
    if current_step < len(data['positions']):
        current_pos = data['positions'][current_step]
        current_vel = data['velocities'][current_step]
        current_time = data['times'][current_step]
        
        # Mission status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mission Time", f"{current_time/3600:.1f} hours")
        
        with col2:
            distance_from_earth = np.linalg.norm(current_pos) - 6371
            st.metric("Altitude", f"{distance_from_earth:.0f} km")
        
        with col3:
            velocity = np.linalg.norm(current_vel)
            st.metric("Speed", f"{velocity:.2f} km/s")
        
        with col4:
            progress_percent = (current_step / len(data['positions'])) * 100
            st.metric("Mission Progress", f"{progress_percent:.1f}%")
        
        # Progress bar
        st.progress(current_step / len(data['positions']))
        
        # Auto-refresh if animation is running
        if st.session_state.animation_running:
            time.sleep(0.1)  # Small delay to control animation speed
            st.rerun()
    
    # Animation information
    st.info(f"""
    **Animation Settings:**
    - Duration: {animation_duration} {time_unit}
    - Total trajectory steps: {len(data['positions'])}
    - Current step: {current_step + 1}
    - Time scale: 1 real second = {data['times'][-1]/total_animation_seconds:.1f} simulation seconds
    """)

def display_eulers_method_details(data):
    st.header("Euler's Method Implementation")
    
    # Display the differential equation
    st.subheader("Differential Equations")
    st.latex(r"""
    \frac{d\vec{r}}{dt} = \vec{v}
    """)
    st.latex(r"""
    \frac{d\vec{v}}{dt} = \vec{a} = \frac{\vec{F}_{total}}{m}
    """)
    
    # Force equations
    st.subheader("Gravitational Forces")
    st.latex(r"""
    \vec{F}_{Earth} = -\frac{GM_E m}{|\vec{r}_{Earth}|^3} \vec{r}_{Earth}
    """)
    st.latex(r"""
    \vec{F}_{Moon} = -\frac{GM_M m}{|\vec{r}_{Moon}|^3} \vec{r}_{Moon}
    """)
    
    # Euler's method formula
    st.subheader("Euler's Method Formula")
    st.latex(r"""
    \vec{r}_{n+1} = \vec{r}_n + \vec{v}_n \Delta t
    """)
    st.latex(r"""
    \vec{v}_{n+1} = \vec{v}_n + \vec{a}_n \Delta t
    """)
    
    # Initial conditions
    st.subheader("Initial Conditions")
    initial_data = {
        'Parameter': ['Position X', 'Position Y', 'Velocity X', 'Velocity Y', 'Step Size', 'Mass'],
        'Value': [
            f"{data['positions'][0][0]:.2f} km",
            f"{data['positions'][0][1]:.2f} km",
            f"{data['velocities'][0][0]:.3f} km/s",
            f"{data['velocities'][0][1]:.3f} km/s",
            f"{data['step_size']:.0f} s",
            f"{data['mass']:.0f} kg"
        ],
        'Description': [
            'Initial X coordinate from Earth center',
            'Initial Y coordinate from Earth center',
            'Initial velocity in X direction',
            'Initial velocity in Y direction',
            'Time step for numerical integration',
            'Spacecraft mass'
        ]
    }
    st.dataframe(pd.DataFrame(initial_data), use_container_width=True)

def display_step_calculations(data):
    st.header("Step-by-Step Calculations")
    
    # Mission phases
    phases = identify_mission_phases(data)
    
    for phase_name, phase_data in phases.items():
        st.subheader(f"🚀 {phase_name}")
        
        # Display first 10 steps of this phase
        start_idx = phase_data['start_idx']
        end_idx = min(start_idx + 10, phase_data['end_idx'])
        
        steps_data = []
        for i in range(start_idx, end_idx):
            # Calculate forces at this step
            forces = calculate_forces_at_step(data, i)
            
            step_info = {
                'Step': i + 1,
                'Time (s)': f"{data['times'][i]:.0f}",
                'Position X (km)': f"{data['positions'][i][0]:.2f}",
                'Position Y (km)': f"{data['positions'][i][1]:.2f}",
                'Velocity X (km/s)': f"{data['velocities'][i][0]:.4f}",
                'Velocity Y (km/s)': f"{data['velocities'][i][1]:.4f}",
                'Force X (N)': f"{forces[0]:.2e}",
                'Force Y (N)': f"{forces[1]:.2e}",
                'Distance from Earth (km)': f"{np.linalg.norm(data['positions'][i]):.2f}",
                'Speed (km/s)': f"{np.linalg.norm(data['velocities'][i]):.4f}"
            }
            steps_data.append(step_info)
        
        df = pd.DataFrame(steps_data)
        st.dataframe(df, use_container_width=True)
        
        # Show the calculation for one step in detail
        if len(steps_data) > 0:
            with st.expander(f"Detailed calculation for step {start_idx + 1}"):
                show_detailed_step_calculation(data, start_idx)

def display_custom_point_analysis(data):
    st.header("Custom Point Analysis")
    
    # Time selection
    max_time = data['times'][-1] / 3600
    selected_time = st.slider("Select time along trajectory (hours)", 0.0, max_time, max_time/2)
    
    # Find closest data point
    time_seconds = selected_time * 3600
    closest_idx = np.argmin(np.abs(data['times'] - time_seconds))
    
    # Display current state
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current State")
        current_pos = data['positions'][closest_idx]
        current_vel = data['velocities'][closest_idx]
        current_time = data['times'][closest_idx]
        
        state_data = {
            'Parameter': ['Time', 'Position X', 'Position Y', 'Velocity X', 'Velocity Y', 'Distance from Earth', 'Speed'],
            'Value': [
                f"{current_time/3600:.2f} hours",
                f"{current_pos[0]:.2f} km",
                f"{current_pos[1]:.2f} km",
                f"{current_vel[0]:.4f} km/s",
                f"{current_vel[1]:.4f} km/s",
                f"{np.linalg.norm(current_pos):.2f} km",
                f"{np.linalg.norm(current_vel):.4f} km/s"
            ]
        }
        st.dataframe(pd.DataFrame(state_data), use_container_width=True)
    
    with col2:
        st.subheader("Remaining Trajectory")
        remaining_steps = len(data['times']) - closest_idx
        remaining_time = (data['times'][-1] - current_time) / 3600
        
        remaining_data = {
            'Parameter': ['Remaining Steps', 'Remaining Time', 'Final Distance from Earth'],
            'Value': [
                f"{remaining_steps}",
                f"{remaining_time:.2f} hours",
                f"{np.linalg.norm(data['positions'][-1]):.2f} km"
            ]
        }
        st.dataframe(pd.DataFrame(remaining_data), use_container_width=True)
    
    # Show next 10 steps from this point
    st.subheader("Next 10 Steps from Selected Point")
    next_steps_data = []
    for i in range(closest_idx, min(closest_idx + 10, len(data['times']))):
        forces = calculate_forces_at_step(data, i)
        
        step_info = {
            'Step': i + 1,
            'Time (s)': f"{data['times'][i]:.0f}",
            'Position X (km)': f"{data['positions'][i][0]:.2f}",
            'Position Y (km)': f"{data['positions'][i][1]:.2f}",
            'Velocity X (km/s)': f"{data['velocities'][i][0]:.4f}",
            'Velocity Y (km/s)': f"{data['velocities'][i][1]:.4f}",
            'Force X (N)': f"{forces[0]:.2e}",
            'Force Y (N)': f"{forces[1]:.2e}"
        }
        next_steps_data.append(step_info)
    
    df = pd.DataFrame(next_steps_data)
    st.dataframe(df, use_container_width=True)

def display_physics_background():
    st.header("Physics Background")
    
    st.markdown("""
    ### Orbital Mechanics Fundamentals
    
    This simulation uses **Euler's method** to numerically solve the differential equations governing spacecraft motion under gravitational forces.
    
    #### Key Concepts:
    - **Gravitational Force**: The primary force acting on the spacecraft
    - **Orbital Velocity**: The speed needed to maintain orbit
    - **Escape Velocity**: The minimum velocity to escape Earth's gravity
    - **Trans-lunar Injection**: The maneuver to leave Earth orbit toward the Moon
    
    #### Mission Phases:
    1. **Liftoff**: Initial acceleration away from Earth
    2. **Earth Orbit**: Circular or elliptical path around Earth
    3. **Trans-lunar Injection**: Trajectory change toward Moon
    4. **Lunar Orbit**: Circular path around Moon
    5. **Landing**: Controlled descent to Moon surface
    """)

def identify_mission_phases(data):
    """Identify different mission phases based on trajectory data"""
    phases = {}
    
    # Simple phase identification based on distance from Earth
    earth_radius = 6371  # km
    positions = data['positions']
    distances = np.linalg.norm(positions, axis=1)
    
    # Liftoff phase (first 10 steps)
    phases['Liftoff'] = {'start_idx': 0, 'end_idx': min(10, len(positions))}
    
    # Earth orbit phase (low altitude)
    orbit_threshold = earth_radius + 2000  # 2000 km above Earth
    earth_orbit_indices = np.where(distances < orbit_threshold)[0]
    if len(earth_orbit_indices) > 10:
        phases['Earth Orbit'] = {
            'start_idx': max(10, earth_orbit_indices[0]), 
            'end_idx': earth_orbit_indices[-1]
        }
    
    # Trans-lunar injection (medium distance)
    trans_lunar_threshold = earth_radius + 50000  # 50000 km from Earth
    trans_lunar_indices = np.where((distances >= orbit_threshold) & (distances < trans_lunar_threshold))[0]
    if len(trans_lunar_indices) > 0:
        phases['Trans-lunar Injection'] = {
            'start_idx': trans_lunar_indices[0],
            'end_idx': trans_lunar_indices[-1]
        }
    
    # Lunar approach (high distance)
    lunar_indices = np.where(distances >= trans_lunar_threshold)[0]
    if len(lunar_indices) > 0:
        phases['Lunar Approach'] = {
            'start_idx': lunar_indices[0],
            'end_idx': len(positions) - 1
        }
    
    return phases

def calculate_forces_at_step(data, step_idx):
    """Calculate gravitational forces at a specific step"""
    physics = st.session_state.physics_engine
    position = data['positions'][step_idx]
    mass = data['mass']
    
    # Earth gravitational force
    earth_force = physics.gravitational_force_earth(position, mass)
    
    # Moon gravitational force
    moon_force = physics.gravitational_force_moon(position, mass, data['times'][step_idx])
    
    # Total force
    total_force = earth_force + moon_force
    
    return total_force

def show_detailed_step_calculation(data, step_idx):
    """Show detailed calculation for a specific step"""
    if step_idx >= len(data['positions']) - 1:
        st.write("No next step available")
        return
    
    # Current state
    r_n = data['positions'][step_idx]
    v_n = data['velocities'][step_idx]
    dt = data['step_size']
    
    # Calculate forces and acceleration
    forces = calculate_forces_at_step(data, step_idx)
    acceleration = forces / data['mass']
    
    # Next state
    r_n_plus_1 = data['positions'][step_idx + 1]
    v_n_plus_1 = data['velocities'][step_idx + 1]
    
    st.write("**Current State:**")
    st.write(f"Position: r_n = [{r_n[0]:.2f}, {r_n[1]:.2f}] km")
    st.write(f"Velocity: v_n = [{v_n[0]:.4f}, {v_n[1]:.4f}] km/s")
    st.write(f"Time step: Δt = {dt:.0f} s")
    
    st.write("**Forces and Acceleration:**")
    st.write(f"Total Force: F = [{forces[0]:.2e}, {forces[1]:.2e}] N")
    st.write(f"Acceleration: a = [{acceleration[0]:.6f}, {acceleration[1]:.6f}] km/s²")
    
    st.write("**Euler's Method Application:**")
    st.write(f"r_(n+1) = r_n + v_n * Δt")
    st.write(f"r_(n+1) = [{r_n[0]:.2f}, {r_n[1]:.2f}] + [{v_n[0]:.4f}, {v_n[1]:.4f}] * {dt:.0f}")
    st.write(f"r_(n+1) = [{r_n_plus_1[0]:.2f}, {r_n_plus_1[1]:.2f}] km")
    
    st.write(f"v_(n+1) = v_n + a_n * Δt")
    st.write(f"v_(n+1) = [{v_n[0]:.4f}, {v_n[1]:.4f}] + [{acceleration[0]:.6f}, {acceleration[1]:.6f}] * {dt:.0f}")
    st.write(f"v_(n+1) = [{v_n_plus_1[0]:.4f}, {v_n_plus_1[1]:.4f}] km/s")

if __name__ == "__main__":
    main()
