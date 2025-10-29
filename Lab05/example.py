import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# Define the state space (words or silence)
states = ["Silence", "Ana", "Are", "Mere"]
n_states = len(states)

# Define the observation space (volume of the speech)
observations = ["Loud", "Soft"]
n_observations = len(observations)

# Updated initial state distribution: adjusted to reduce the dominance of "Silence"
start_probability = np.array([0.4, 0.2, 0.2, 0.2])
# Explanation:
# Starting in "Silence" is less dominant now, making it possible to begin with any word.

# Updated state transition probabilities for a more balanced progression
transition_probability = np.array([
    [0.5, 0.3, 0.1, 0.1],  # Silence: more likely to move to "Ana" or stay in Silence
    [0.2, 0.2, 0.5, 0.1],  # Ana: more likely to transition to "Are"
    [0.1, 0.2, 0.2, 0.5],  # Are: likely to transition to "Mere"
    [0.3, 0.1, 0.1, 0.5]   # Mere: can go back to Silence or stay on "Mere"
])
# Explanation:
# This setup allows the model to naturally transition from "Silence" to words, and from one word to the next,
# while still allowing some chance of returning to previous words or silence.

# Define the observation likelihoods (emission probabilities)
emission_probability = np.array([
    [0.8, 0.2],  # Silence: still more likely to be "Soft"
    [0.3, 0.7],  # Ana: more likely to be "Loud"
    [0.7, 0.3],  # Are: most likely to be "Loud"
    [0.4, 0.6]   # Mere: more likely to be "Soft"
])
# Explanation:
# The probabilities are adjusted to reflect each word's likelihood of being "Loud" or "Soft."

# Create and configure the HMM model
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Define a sequence of observations (observed volume levels over time)
observations_sequence = np.array([1, 0, 1, 1, 0, 0, 1, 0]).reshape(-1, 1)
# Explanation:
# This sequence provides a mix of "Loud" (0) and "Soft" (1) observations to test how the model interprets the pattern.

# Predict the most likely hidden states (words or silence) given the observation sequence
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states:", hidden_states)

# Plot the results for visualization
sns.set_style("darkgrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.xlabel("Time Step")
plt.ylabel("Hidden State (Word or Silence)")
plt.yticks(ticks=range(n_states), labels=states)
plt.legend()
plt.title("Predicted Hidden States Over Time")
plt.show()
