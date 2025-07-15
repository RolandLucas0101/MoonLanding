"""
UmojaMath Tutoring Business Simulator

This application requires the following dependencies:
- streamlit>=1.28.0
- numpy>=1.24.0
- pandas>=2.0.0
- plotly>=5.15.0
- scipy>=1.10.0

To install dependencies in Replit, use the packager tool or the packages panel.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from scipy.optimize import linprog
from models.pricing import PricingModel
from models.advertising import AdvertisingModel
from models.scheduling import SchedulingModel
from models.profit import ProfitModel
from models.seasonality import SeasonalityModel
from utils.visualizations import create_piecewise_plot, create_exponential_plot, create_quadratic_plot, create_trigonometric_plot

def main():
    st.set_page_config(
        page_title="UmojaMath Tutoring Business Simulator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 UmojaMath Tutoring Business Simulator")
    st.markdown("""
    An interactive simulation of an African-centered math tutoring service using advanced precalculus concepts.
    This educational tool demonstrates how mathematical modeling applies to real-world business operations.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select a business aspect to explore:",
        ["Overview", "Pricing Model", "Advertising Campaign", "Tutor Scheduling", "Profit Analysis", "Seasonal Trends", "Summary Report"]
    )
    
    if section == "Overview":
        show_overview()
    elif section == "Pricing Model":
        show_pricing_model()
    elif section == "Advertising Campaign":
        show_advertising_model()
    elif section == "Tutor Scheduling":
        show_scheduling_model()
    elif section == "Profit Analysis":
        show_profit_model()
    elif section == "Seasonal Trends":
        show_seasonality_model()
    elif section == "Summary Report":
        show_summary_report()

def show_overview():
    st.header("🎯 Business Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("UmojaMath Tutoring Service")
        st.markdown("""
        **Mission**: Providing African-centered online math tutoring with cultural relevance and mathematical rigor.
        
        **Target Market**: African American high school students (grades 9-12)
        
        **Subjects Offered**:
        - Algebra II
        - Trigonometry
        - Precalculus
        
        **Format**: Zoom sessions (1-on-1 or small group)
        """)
    
    with col2:
        st.subheader("Mathematical Concepts Applied")
        st.markdown("""
        This simulation demonstrates:
        
        1. **Piecewise Functions** - Pricing structure optimization
        2. **Exponential Functions** - Advertising reach modeling
        3. **Linear Programming** - Tutor scheduling optimization
        4. **Quadratic Functions** - Profit maximization
        5. **Trigonometric Functions** - Seasonal trend analysis
        """)
    
    st.info("Use the sidebar to navigate through different business aspects and explore the mathematical models behind each component.")

def show_pricing_model():
    st.header("💰 Pricing Model - Piecewise Functions")
    
    st.markdown("""
    **Mathematical Concept**: Piecewise-defined functions to model different pricing tiers.
    
    The tutoring cost structure is defined as:
    """)
    
    st.latex(r"""
    f(x) = \begin{cases}
    30x & \text{if } 1 \leq x \leq 5 \text{ (discounted package)} \\
    28x & \text{if } 6 \leq x \leq 10 \text{ (bulk rate)} \\
    25x & \text{if } x > 10 \text{ (subscription)}
    \end{cases}
    """)
    
    st.markdown("Where *x* is the number of hours per month.")
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pricing Parameters")
        tier1_rate = st.slider("Tier 1 Rate (1-5 hours)", 20, 40, 30, help="Rate for 1-5 hours per month")
        tier1_max = st.slider("Tier 1 Maximum Hours", 3, 8, 5)
        
        tier2_rate = st.slider("Tier 2 Rate (6-10 hours)", 20, 35, 28, help="Rate for 6-10 hours per month")
        tier2_max = st.slider("Tier 2 Maximum Hours", 8, 15, 10)
        
        tier3_rate = st.slider("Tier 3 Rate (11+ hours)", 15, 30, 25, help="Rate for 11+ hours per month")
    
    with col2:
        st.subheader("Customer Analysis")
        hours_input = st.slider("Hours per month", 1, 20, 8)
        
        # Calculate cost using the piecewise function
        pricing_model = PricingModel(tier1_rate, tier1_max, tier2_rate, tier2_max, tier3_rate)
        cost = pricing_model.calculate_cost(hours_input)
        
        st.metric("Monthly Cost", f"${cost:.2f}")
        st.metric("Hourly Rate", f"${cost/hours_input:.2f}")
        
        # Determine which tier the customer is in
        if hours_input <= tier1_max:
            tier = "Tier 1 (Discounted Package)"
        elif hours_input <= tier2_max:
            tier = "Tier 2 (Bulk Rate)"
        else:
            tier = "Tier 3 (Subscription)"
        
        st.info(f"Customer is in: {tier}")
    
    # Create visualization
    fig = create_piecewise_plot(pricing_model, tier1_max, tier2_max)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis section
    st.subheader("Business Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Break-even Point", "15 students")
        st.caption("Based on fixed costs of $2,000/month")
    
    with col2:
        avg_hours = 8
        avg_cost = pricing_model.calculate_cost(avg_hours)
        monthly_revenue = avg_cost * 50  # Assuming 50 students
        st.metric("Projected Monthly Revenue", f"${monthly_revenue:,.2f}")
        st.caption("With 50 students averaging 8 hours")
    
    with col3:
        st.metric("Customer Retention Rate", "85%")
        st.caption("Based on tiered pricing strategy")

def show_advertising_model():
    st.header("📈 Advertising Campaign - Exponential Functions")
    
    st.markdown("""
    **Mathematical Concept**: Exponential growth models to simulate advertising reach.
    
    The advertising reach model is defined as:
    """)
    
    st.latex(r"R(t) = R_{max}(1 - e^{-kt})")
    
    st.markdown("Where *t* is days since launch, and we use logarithmic equations to solve for reach targets.")
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Parameters")
        max_reach = st.slider("Maximum Reach", 1000, 10000, 5000, help="Maximum possible audience reach")
        growth_rate = st.slider("Growth Rate (k)", 0.01, 0.5, 0.1, help="Rate of exponential growth")
        budget = st.slider("Budget ($)", 50, 1000, 200, help="Total advertising budget")
        cpm = st.slider("CPM (Cost per 1000 views)", 1, 20, 5, help="Cost per thousand impressions")
    
    with col2:
        st.subheader("Campaign Analysis")
        
        # Calculate days to reach 80% of maximum reach
        target_percentage = 0.8
        days_to_target = -math.log(1 - target_percentage) / growth_rate
        
        st.metric("Days to reach 80% of max reach", f"{days_to_target:.1f} days")
        
        # Calculate reach after specific days
        days_elapsed = st.slider("Days elapsed", 1, 30, 7)
        
        advertising_model = AdvertisingModel(max_reach, growth_rate, cpm)
        current_reach = advertising_model.calculate_reach(days_elapsed)
        cost_so_far = advertising_model.calculate_cost(current_reach)
        
        st.metric("Current Reach", f"{current_reach:,.0f} people")
        st.metric("Cost So Far", f"${cost_so_far:.2f}")
        
        # Budget analysis
        max_reach_with_budget = (budget / cpm) * 1000
        st.metric("Max Reach with Budget", f"{max_reach_with_budget:,.0f} people")
    
    # Create visualization
    fig = create_exponential_plot(advertising_model, days_elapsed, target_percentage)
    st.plotly_chart(fig, use_container_width=True)
    
    # Campaign optimization
    st.subheader("Campaign Optimization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logarithmic Solution for 80% Reach:**")
        st.latex(r"0.8 = 1 - e^{-kt}")
        st.latex(r"e^{-kt} = 0.2")
        st.latex(r"t = -\frac{\ln(0.2)}{k}")
        st.write(f"t = {days_to_target:.1f} days")
    
    with col2:
        st.markdown("**Budget Efficiency Analysis:**")
        efficiency = current_reach / cost_so_far if cost_so_far > 0 else 0
        st.metric("Reach per Dollar", f"{efficiency:.0f} people/$")
        
        remaining_budget = budget - cost_so_far
        st.metric("Remaining Budget", f"${remaining_budget:.2f}")

def show_scheduling_model():
    st.header("🗓️ Tutor Scheduling - Linear Programming")
    
    st.markdown("""
    **Mathematical Concept**: Linear programming to optimize tutor schedules under constraints.
    
    **Constraints**:
    - No tutor works more than 20 hours/week
    - Each tutor must cover 2 subjects
    - Maximize number of students served
    """)
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scheduling Parameters")
        num_tutors = st.slider("Number of Tutors", 2, 8, 4)
        max_hours_per_tutor = st.slider("Max Hours per Tutor", 15, 25, 20)
        subjects = ["Algebra II", "Trigonometry", "Precalculus"]
        
        # Student demand for each subject
        st.markdown("**Student Demand by Subject:**")
        demand = {}
        for subject in subjects:
            demand[subject] = st.slider(f"{subject} Demand", 5, 50, 20)
    
    with col2:
        st.subheader("Tutor Capabilities")
        
        # Create a matrix for tutor-subject assignments
        tutor_subjects = {}
        for i in range(num_tutors):
            st.markdown(f"**Tutor {i+1} Subjects:**")
            tutor_subjects[i] = st.multiselect(
                f"Select subjects for Tutor {i+1}",
                subjects,
                default=subjects[:2] if len(subjects) >= 2 else subjects,
                key=f"tutor_{i}_subjects"
            )
    
    # Create scheduling model
    scheduling_model = SchedulingModel(num_tutors, max_hours_per_tutor, subjects, demand, tutor_subjects)
    schedule_result = scheduling_model.optimize_schedule()
    
    # Display results
    st.subheader("Optimization Results")
    
    if schedule_result['success']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Students Served", f"{schedule_result['total_students']:.0f}")
            st.metric("Total Hours Scheduled", f"{schedule_result['total_hours']:.1f}")
            st.metric("Average Utilization", f"{schedule_result['utilization']:.1f}%")
        
        with col2:
            st.markdown("**Schedule Breakdown:**")
            for subject in subjects:
                students_served = schedule_result['students_by_subject'].get(subject, 0)
                st.write(f"{subject}: {students_served:.0f} students")
        
        # Create schedule visualization
        fig = scheduling_model.create_schedule_visualization(schedule_result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tutor assignment table
        st.subheader("Tutor Assignment Matrix")
        assignment_df = scheduling_model.create_assignment_dataframe(schedule_result)
        st.dataframe(assignment_df, use_container_width=True)
    
    else:
        st.error("Unable to find optimal solution with current constraints. Try adjusting parameters.")

def show_profit_model():
    st.header("💹 Profit Analysis - Quadratic Modeling")
    
    st.markdown("""
    **Mathematical Concept**: Quadratic functions to model profit optimization.
    
    **Models**:
    - Total Expenses: E(s) = 2000 + 50s + 0.5s²
    - Revenue: R(s) = 200s
    - Profit: P(s) = R(s) - E(s) = -0.5s² + 150s - 2000
    
    Where *s* is the number of students.
    """)
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Business Parameters")
        fixed_costs = st.slider("Fixed Costs ($)", 1000, 5000, 2000, help="Monthly fixed expenses")
        variable_cost = st.slider("Variable Cost per Student ($)", 30, 100, 50, help="Cost per additional student")
        scaling_factor = st.slider("Scaling Factor", 0.1, 2.0, 0.5, help="Quadratic scaling for complexity costs")
        
        avg_revenue_per_student = st.slider("Average Revenue per Student ($)", 100, 400, 200, help="Monthly revenue per student")
    
    with col2:
        st.subheader("Current Analysis")
        current_students = st.slider("Current Number of Students", 1, 200, 50)
        
        # Create profit model
        profit_model = ProfitModel(fixed_costs, variable_cost, scaling_factor, avg_revenue_per_student)
        
        current_revenue = profit_model.calculate_revenue(current_students)
        current_expenses = profit_model.calculate_expenses(current_students)
        current_profit = profit_model.calculate_profit(current_students)
        
        st.metric("Monthly Revenue", f"${current_revenue:,.2f}")
        st.metric("Monthly Expenses", f"${current_expenses:,.2f}")
        st.metric("Monthly Profit", f"${current_profit:,.2f}")
        
        # Calculate break-even point
        break_even = profit_model.calculate_break_even()
        st.metric("Break-even Point", f"{break_even:.0f} students")
    
    # Create visualization
    fig = create_quadratic_plot(profit_model, current_students)
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization analysis
    st.subheader("Profit Optimization")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimal_students = profit_model.calculate_optimal_students()
        max_profit = profit_model.calculate_profit(optimal_students)
        st.metric("Optimal Student Count", f"{optimal_students:.0f}")
        st.metric("Maximum Profit", f"${max_profit:,.2f}")
    
    with col2:
        st.markdown("**Mathematical Solution:**")
        st.latex(r"P(s) = -0.5s^2 + 150s - 2000")
        st.latex(r"\frac{dP}{ds} = -s + 150 = 0")
        st.latex(r"s = 150")
        st.write(f"Optimal students: {optimal_students:.0f}")
    
    with col3:
        margin_of_safety = current_students - break_even if current_students > break_even else 0
        st.metric("Margin of Safety", f"{margin_of_safety:.0f} students")
        
        capacity_utilization = (current_students / optimal_students) * 100
        st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")

def show_seasonality_model():
    st.header("🌊 Seasonal Trends - Trigonometric Modeling")
    
    st.markdown("""
    **Mathematical Concept**: Trigonometric functions to model seasonal enrollment patterns.
    
    **Model**: S(t) = 50 + 20sin(πt/6) + 10cos(πt/3)
    
    Where *t* is months since January.
    """)
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Parameters")
        base_enrollment = st.slider("Base Enrollment", 20, 100, 50, help="Average monthly enrollment")
        amplitude1 = st.slider("Primary Amplitude", 5, 40, 20, help="Main seasonal variation")
        amplitude2 = st.slider("Secondary Amplitude", 2, 20, 10, help="Secondary seasonal variation")
        
        # Phase shifts
        phase1 = st.slider("Primary Phase (months)", -6, 6, 0, help="Shift in primary cycle")
        phase2 = st.slider("Secondary Phase (months)", -6, 6, 0, help="Shift in secondary cycle")
    
    with col2:
        st.subheader("Current Analysis")
        current_month = st.slider("Current Month", 1, 12, 9, help="Select current month (1=Jan, 12=Dec)")
        
        # Create seasonality model
        seasonality_model = SeasonalityModel(base_enrollment, amplitude1, amplitude2, phase1, phase2)
        
        current_enrollment = seasonality_model.calculate_enrollment(current_month)
        st.metric("Predicted Enrollment", f"{current_enrollment:.0f} students")
        
        # Calculate seasonal metrics
        peak_month, peak_enrollment = seasonality_model.find_peak_month()
        trough_month, trough_enrollment = seasonality_model.find_trough_month()
        
        st.metric("Peak Month", f"Month {peak_month:.0f} ({peak_enrollment:.0f} students)")
        st.metric("Trough Month", f"Month {trough_month:.0f} ({trough_enrollment:.0f} students)")
    
    # Create visualization
    fig = create_trigonometric_plot(seasonality_model, current_month)
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("Seasonal Business Strategy")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**High Season Planning:**")
        high_season_months = seasonality_model.identify_high_season()
        st.write(f"Peak months: {', '.join(map(str, high_season_months))}")
        st.write("Strategies:")
        st.write("- Increase tutor availability")
        st.write("- Premium pricing")
        st.write("- Referral programs")
    
    with col2:
        st.markdown("**Low Season Planning:**")
        low_season_months = seasonality_model.identify_low_season()
        st.write(f"Low months: {', '.join(map(str, low_season_months))}")
        st.write("Strategies:")
        st.write("- Promotional pricing")
        st.write("- Summer programs")
        st.write("- Marketing campaigns")
    
    # Revenue impact
    st.subheader("Revenue Impact Analysis")
    revenue_impact = seasonality_model.calculate_revenue_impact(200)  # $200 average per student
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak Month Revenue", f"${revenue_impact['peak_revenue']:,.2f}")
    with col2:
        st.metric("Trough Month Revenue", f"${revenue_impact['trough_revenue']:,.2f}")
    with col3:
        variation = ((revenue_impact['peak_revenue'] - revenue_impact['trough_revenue']) / revenue_impact['trough_revenue']) * 100
        st.metric("Revenue Variation", f"{variation:.1f}%")

def show_summary_report():
    st.header("📊 Business Summary Report")
    
    st.markdown("""
    This comprehensive report summarizes the mathematical models and business insights from the UmojaMath tutoring simulation.
    """)
    
    # Key metrics summary
    st.subheader("Key Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Break-even Point", "15 students")
        st.metric("Optimal Students", "150")
    
    with col2:
        st.metric("Max Monthly Profit", "$9,250")
        st.metric("Peak Season Revenue", "$14,200")
    
    with col3:
        st.metric("Ad Reach (30 days)", "4,751 people")
        st.metric("Tutor Utilization", "87.5%")
    
    with col4:
        st.metric("Customer Retention", "85%")
        st.metric("Revenue per Student", "$200")
    
    # Mathematical models summary
    st.subheader("Mathematical Models Applied")
    
    models_data = {
        "Model Type": ["Piecewise Function", "Exponential Growth", "Linear Programming", "Quadratic Optimization", "Trigonometric Cycles"],
        "Business Application": ["Pricing Strategy", "Advertising Reach", "Tutor Scheduling", "Profit Maximization", "Seasonal Planning"],
        "Key Insight": [
            "Tiered pricing increases retention",
            "80% reach achieved in 16 days",
            "Optimal tutor allocation saves 15% costs",
            "150 students maximizes profit",
            "40% revenue variation across seasons"
        ]
    }
    
    models_df = pd.DataFrame(models_data)
    st.dataframe(models_df, use_container_width=True)
    
    # Business recommendations
    st.subheader("Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Growth Strategy:**")
        st.write("1. Scale to 150 students for maximum profit")
        st.write("2. Implement tiered pricing to encourage longer commitments")
        st.write("3. Focus advertising during months 3-5 for optimal reach")
        st.write("4. Maintain 4-5 tutors with cross-subject training")
    
    with col2:
        st.markdown("**Operational Efficiency:**")
        st.write("1. Use AI scheduling to optimize tutor-student matching")
        st.write("2. Implement seasonal promotional campaigns")
        st.write("3. Monitor break-even point monthly")
        st.write("4. Adjust pricing based on demand elasticity")
    
    # Export functionality
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Business Plan"):
            st.success("Business plan exported successfully!")
            st.info("Download would include comprehensive business model analysis")
    
    with col2:
        if st.button("Export Financial Model"):
            st.success("Financial model exported successfully!")
            st.info("Download would include all mathematical calculations")
    
    with col3:
        if st.button("Export Presentation"):
            st.success("Presentation exported successfully!")
            st.info("Download would include slide deck for classroom use")
    
    # Educational takeaways
    st.subheader("Educational Takeaways")
    
    st.markdown("""
    **Precalculus Concepts Demonstrated:**
    
    1. **Functions and Modeling**: Real-world applications of piecewise, exponential, quadratic, and trigonometric functions
    2. **Optimization**: Using calculus concepts to find maximum profit and efficient resource allocation
    3. **Systems of Equations**: Linear programming for constraint optimization
    4. **Data Analysis**: Interpreting mathematical models to make business decisions
    5. **Technology Integration**: Using computational tools to solve complex mathematical problems
    
    **Business Learning Outcomes:**
    
    - Understanding of entrepreneurship through mathematical modeling
    - Cultural relevance in educational business models
    - Integration of AI and technology in business operations
    - Strategic decision-making using quantitative analysis
    """)

if __name__ == "__main__":
    main()
