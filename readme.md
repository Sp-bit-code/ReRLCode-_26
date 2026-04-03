# Smart Predictive Irrigation using Q-Learning

## What is this project

This project is an intelligent irrigation system that uses Reinforcement Learning (Q-Learning) to decide how much water should be given to crops based on environmental conditions.

The system learns from historical sensor data such as soil moisture, temperature, humidity, wind speed, solar radiation (PAR), and evapotranspiration (ETo). It trains an agent that can take optimal irrigation decisions to maintain soil moisture in a desired range while minimizing water usage.

---

## Problem Statement

Traditional irrigation methods either over-water or under-water crops because they do not adapt dynamically to changing weather and soil conditions.

This leads to:

* Water wastage
* Reduced crop yield
* Inefficient resource usage

This project solves this problem by learning an optimal irrigation policy using data-driven decision making.

---

## How it works

1. Data Preprocessing
   Sensor data is cleaned and transformed into meaningful features such as:

   * Soil moisture
   * Temperature
   * Humidity
   * Wind speed
   * Solar radiation
   * Evapotranspiration

2. Environment Simulation
   A custom irrigation environment is created where:

   * State = current environmental conditions
   * Action = amount of water (in mm)
   * Reward = based on how close soil moisture stays within the optimal range

3. Q-Learning Agent
   The agent learns using:

   * State-action Q-table
   * Epsilon-greedy exploration
   * Reward-based updates

   It improves over episodes and learns the best irrigation strategy.

4. Training
   The agent interacts with the environment over multiple episodes and updates Q-values using:

   * Learning rate
   * Discount factor
   * Exploration decay

5. Policy Learning
   After training, the agent learns:

   * When to irrigate
   * How much water to apply
   * How to balance water usage and crop needs

6. Deployment (Streamlit Interface)
   A user-friendly interface allows:

   * Training visualization
   * Episode-wise analysis
   * Live agent behavior tracking
   * Smart irrigation plan generation

---

## Key Features

* Reinforcement Learning based decision system
* Custom irrigation environment simulation
* Dynamic reward function based on soil moisture balance
* Episode-wise and step-wise agent logging
* Q-table visualization and policy analysis
* Smart 10-day irrigation planner based on user input
* Water usage optimization and savings calculation
* Streamlit-based interactive dashboard

---

## Input Features

The model uses the following inputs:

* Soil moisture
* Temperature
* Humidity
* Wind speed
* Solar radiation (PAR)
* Evapotranspiration (ETo)
* Rain probability (simulated)

---

## Output

* Optimal irrigation action (in mm)
* Daily irrigation schedule
* Soil moisture trend
* Total water usage
* Water savings compared to baseline

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Streamlit
* Reinforcement Learning (Q-Learning)

---

## Project Structure

```
irrigation_rl_project/
│
├── preprocess.py          # Data cleaning and feature engineering
├── environment.py         # Custom irrigation environment
├── rl_agent.py            # Q-Learning agent implementation
├── app.py                 # Streamlit dashboard
│
├── artifacts/
│   └── q_table.pkl        # Saved trained model
│
└── data/
    └── sensor datasets
```

---

## Results

* The agent learns optimal irrigation policies over time
* Avoids over-irrigation and under-irrigation
* Improves soil moisture stability
* Achieves higher reward compared to baseline (no irrigation strategy)
* Reduces unnecessary water usage

---

## Future Improvements

* Integrate real-time IoT sensor data
* Replace Q-learning with Deep Reinforcement Learning (DQN / PPO)
* Add crop-specific models
* Use weather API for real forecasts
* Deploy on edge devices for smart farming

---

## Conclusion

This project demonstrates how Reinforcement Learning can be used to solve real-world agricultural problems. The system learns optimal irrigation strategies that improve efficiency, reduce water waste, and support sustainable farming.
