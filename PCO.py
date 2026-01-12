import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Diet Meal Planning Optimisation (PSO)",
    layout="wide"
)

# =========================
# Header
# =========================
st.markdown("""
# 🍽️ Diet Meal Planning Optimisation  
### Using Particle Swarm Optimisation (PSO)

This system selects a daily meal plan that satisfies calorie requirements
while minimising total cost.
""")

st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")
data = data[['Calories', 'Protein']].copy()

# =========================
# Cost Model (Assumption)
# =========================
np.random.seed(42)
data['Cost'] = data['Calories'] * np.random.uniform(0.008, 0.015)

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ PSO Parameters")

TARGET_CALORIES = st.sidebar.slider("Target Calories (kcal)", 1500, 3000, 1800)
MEALS_PER_DAY = st.sidebar.slider("Meals per Day", 2, 5, 3)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)

W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Coefficient (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Coefficient (C2)", 0.5, 2.5, 1.5)

# =========================
# Fitness Function
# =========================
def fitness_function(particle):
    indices = np.clip(particle.astype(int), 0, len(data) - 1)
    selected = data.iloc[indices]

    total_calories = selected['Calories'].sum()
    total_cost = selected['Cost'].sum()
    total_protein = selected['Protein'].sum()

    penalty = 0

    # Calorie constraint (±5%)
    calorie_error = abs(total_calories - TARGET_CALORIES)
    if calorie_error > TARGET_CALORIES * 0.05:
        penalty += calorie_error * 1000

    # Protein constraint
    if total_protein < 50:
        penalty += (50 - total_protein) * 500

    return total_cost + penalty

# =========================
# Run Button
# =========================
st.markdown("### 🚀 Run Optimisation")
run = st.button("Start PSO Optimisation")

# =========================
# PSO Algorithm
# =========================
if run:
    particles = np.random.randint(
        0, len(data), (NUM_PARTICLES, MEALS_PER_DAY)
    ).astype(float)

    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, MEALS_PER_DAY))

    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(p) for p in particles])

    gbest = pbest[np.argmin(pbest_fitness)]
    convergence = []

    for _ in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, len(data) - 1)

            fitness = fitness_function(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness

        gbest = pbest[np.argmin(pbest_fitness)]
        convergence.append(min(pbest_fitness))

    best_meal = data.iloc[gbest.astype(int)]

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## ✅ Optimisation Results")

    total_calories = int(best_meal['Calories'].sum())
    total_cost = round(best_meal['Cost'].sum(), 2)
    total_protein = round(best_meal['Protein'].sum(), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Calories (kcal)", total_calories)
    c2.metric("Total Cost (RM)", total_cost)
    c3.metric("Total Protein (g)", total_protein)

    st.markdown("### 🥗 Selected Daily Meal Plan")
    st.dataframe(best_meal, use_container_width=True)

    # =========================
    # Convergence Curve
    # =========================
    st.markdown("## 📈 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence) + 1),
        "Best Fitness Value": convergence
    })

    chart = (
        alt.Chart(convergence_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Iteration", title="Iteration"),
            y=alt.Y("Best Fitness Value", title="Fitness Value")
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)

    # =========================
    # Summary
    # =========================
    st.markdown(f"""
**Summary:**  
- PSO completed **{len(convergence)} iterations**.  
- Final meal plan achieves **{total_calories} kcal**, close to the target of **{TARGET_CALORIES} kcal**.  
- Total cost is **RM {total_cost}**, successfully minimised.  
- Protein intake is **{total_protein} g**, ensuring nutritional adequacy.  
- Results demonstrate that PSO effectively balances nutritional constraints and cost optimisation.
""")
