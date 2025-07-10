# Moon Launch Trajectory Simulation

## Overview

This is an educational space trajectory simulation application built with Streamlit that demonstrates orbital mechanics using Euler's method for numerical integration. The application simulates spacecraft trajectories from Earth to the Moon, providing an interactive learning environment for understanding orbital dynamics, gravitational forces, and trajectory planning.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with separation of concerns:

### Frontend Architecture
- **Streamlit Web Interface**: Interactive web application using Streamlit for the user interface
- **Plotly Visualizations**: Interactive charts and graphs for trajectory visualization
- **Session State Management**: Streamlit's session state handles application state persistence

### Backend Architecture
- **Physics Engine**: Core physics calculations for gravitational forces and orbital mechanics
- **Trajectory Calculator**: Numerical integration using Euler's method for trajectory computation
- **Visualization Module**: Specialized plotting and data visualization components

### Component Structure
```
app.py                    # Main Streamlit application
├── physics_engine.py     # Physics calculations and constants
├── trajectory_calculator.py  # Euler's method integration
└── visualization.py      # Interactive plotting and charts
```

## Key Components

### 1. Physics Engine (`physics_engine.py`)
- **Purpose**: Handles all physics calculations including gravitational forces
- **Key Features**:
  - Earth and Moon gravitational force calculations
  - Physical constants (G, planetary masses, radii)
  - Unit conversions between meters and kilometers
  - Time-dependent Moon position calculations

### 2. Trajectory Calculator (`trajectory_calculator.py`)
- **Purpose**: Implements Euler's method for numerical integration
- **Key Features**:
  - Step-by-step trajectory calculation
  - Configurable time steps and simulation duration
  - Position, velocity, and acceleration tracking
  - Spacecraft mass consideration

### 3. Visualization Module (`visualization.py`)
- **Purpose**: Creates interactive visualizations of trajectory data
- **Key Features**:
  - Multi-panel dashboard with trajectory plots
  - Real-time parameter visualization
  - Interactive Plotly charts
  - Earth and Moon position rendering

### 4. Main Application (`app.py`)
- **Purpose**: Streamlit interface and application orchestration
- **Key Features**:
  - Parameter input controls (sliders, selectboxes)
  - Real-time trajectory calculation
  - Session state management
  - Responsive web interface

## Data Flow

1. **User Input**: Parameters entered through Streamlit sidebar controls
2. **Physics Setup**: Physics engine initialized with constants and spacecraft mass
3. **Trajectory Calculation**: Euler's method integration calculates position/velocity over time
4. **Visualization**: Results processed and displayed through interactive Plotly charts
5. **Session Persistence**: Results stored in Streamlit session state for reuse

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **NumPy**: Numerical computing for physics calculations and array operations
- **Matplotlib**: Basic plotting capabilities (legacy support)
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization and charting

### Physics Dependencies
- Built-in physics constants and calculations
- No external physics libraries required
- Self-contained orbital mechanics implementation

## Deployment Strategy

### Development Environment
- **Platform**: Replit-ready Python environment
- **Dependencies**: All libraries available through pip
- **Configuration**: Streamlit runs directly with `streamlit run app.py`

### Production Considerations
- **Hosting**: Streamlit Cloud or similar platform deployment
- **Performance**: Optimized for educational use cases
- **Scalability**: Single-user sessions with session state management

### Architecture Decisions

1. **Streamlit Choice**: Selected for rapid prototyping and educational accessibility
   - **Pros**: Quick development, built-in interactivity, minimal setup
   - **Cons**: Limited to single-user sessions, less customization than full web frameworks

2. **Euler's Method**: Chosen for educational transparency over accuracy
   - **Pros**: Simple to understand and implement, good for learning
   - **Cons**: Less accurate than Runge-Kutta methods for orbital mechanics

3. **Modular Design**: Separate classes for physics, calculation, and visualization
   - **Pros**: Clean separation of concerns, testable components, reusable code
   - **Cons**: Slightly more complex than monolithic approach

4. **Plotly Visualization**: Interactive charts over static matplotlib
   - **Pros**: Interactive features, professional appearance, web-native
   - **Cons**: Additional dependency, potential performance overhead

The application prioritizes educational value and ease of use over computational accuracy, making it ideal for learning orbital mechanics concepts through hands-on experimentation.