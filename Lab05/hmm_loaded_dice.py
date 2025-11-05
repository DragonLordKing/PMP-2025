import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# HMM Demo: Loaded vs Fair Dice (imperative version)
# - Hidden states: Fair (0), Loaded (1)
# - Observations: die faces 1..6 (internally encoded as 0..5)
#
# Run:
#   python Lab05/hmm_loaded_dice.py

# -----------------------------
# 1) Define the true HMM
# -----------------------------
states = ["Fair", "Loaded"]
n_states = len(states)

faces = [1, 2, 3, 4, 5, 6]
n_faces = len(faces)  # emissions encoded as 0..5

# Start mostly in Fair state
startprob_true = np.array([0.95, 0.05])

# Rare switching: Fair -> Loaded is rare; Loaded tends to persist
transmat_true = np.array([
    [0.95, 0.05],
    [0.10, 0.90],
])

# Emission probabilities: Fair ~ uniform; Loaded strongly favors face '6'
emissionprob_true = np.array([
    [1 / 6.0] * n_faces,            # Fair
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],  # Loaded
])

model_true = hmm.CategoricalHMM(n_components=n_states, random_state=42)
model_true.startprob_ = startprob_true
model_true.transmat_ = transmat_true
model_true.emissionprob_ = emissionprob_true

# -----------------------------
# 2) Sample observations and hidden states
# -----------------------------
num_steps = 200
observations, hidden_states_true = model_true.sample(num_steps)
observations = observations.astype(int)         # shape (T, 1), values in [0, 5]
hidden_states_true = hidden_states_true.astype(int)  # shape (T,), values in {0,1}

print("HMM Loaded Dice Demo (imperative)")
print(f"Sequence length: {num_steps}")

# -----------------------------
# 3) Fit an HMM from observations only (unsupervised)
# -----------------------------
model_learned = hmm.CategoricalHMM(
    n_components=2,
    n_iter=100,
    tol=1e-4,
    random_state=0,
    init_params="ste",  # learn startprob_, transmat_, emissionprob_
)
model_learned.fit(observations)

# -----------------------------
# 4) Decode: most likely state sequence and posterior probabilities
# -----------------------------
predicted_states = model_learned.predict(observations)
posteriors = model_learned.predict_proba(observations)

# Align learned state labels so that index 1 corresponds to Loaded-like
# We decide by the probability of emitting face '6' (index 5)
prob_face_six = model_learned.emissionprob_[:, 5]
loaded_idx = int(np.argmax(prob_face_six))
fair_idx = 1 - loaded_idx

# Remap predicted labels to aligned order: Fair->0, Loaded->1
mapping = {fair_idx: 0, loaded_idx: 1}
predicted_states_aligned = np.vectorize(mapping.get)(predicted_states)

# Compute decoding accuracy vs true hidden states
accuracy = np.mean(predicted_states_aligned == hidden_states_true)
print(f"Decoding accuracy (after label alignment): {accuracy:.3f}")

# Peek at learned parameters in aligned order
print("\nLearned startprob_ (aligned order Fair, Loaded):")
print(np.array([model_learned.startprob_[fair_idx], model_learned.startprob_[loaded_idx]]))

print("\nLearned transmat_ (rows: from, cols: to) in aligned order:")
aligned_trans = model_learned.transmat_[[fair_idx, loaded_idx]][:, [fair_idx, loaded_idx]]
print(aligned_trans)

print("\nLearned emissionprob_ (rows aligned Fair/Loaded; cols faces 1..6):")
aligned_emiss = model_learned.emissionprob_[[fair_idx, loaded_idx]]
print(np.round(aligned_emiss, 3))

# -----------------------------
# 5) Visualize
# -----------------------------
sns.set_style("darkgrid")

time_steps = np.arange(observations.shape[0])

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

# (a) Posterior probability of Loaded over time
axes[0].plot(
    time_steps,
    posteriors[:, loaded_idx],
    label="P(Loaded | observations)",
)
axes[0].set_ylabel("Posterior Probability (Loaded)")
axes[0].set_xlabel("Time Step")
axes[0].set_title("Decoding Hidden Regimes: Loaded vs Fair")

# Shade true Loaded segments for easy comparison
is_true_loaded = hidden_states_true == 1
axes[0].fill_between(
    time_steps,
    0,
    1,
    where=is_true_loaded,
    color="tomato",
    alpha=0.15,
    transform=axes[0].get_xaxis_transform(),
    label="True Loaded regime",
)
axes[0].legend(loc="upper right")

# (b) Emission distributions: True vs Learned
width = 0.18
x_pos = np.arange(6)

true_fair = model_true.emissionprob_[0]
true_loaded = model_true.emissionprob_[1]
learned_fair = model_learned.emissionprob_[fair_idx]
learned_loaded = model_learned.emissionprob_[loaded_idx]

axes[1].bar(x_pos - 1.5 * width, true_fair, width=width, label="True Fair")
axes[1].bar(x_pos - 0.5 * width, learned_fair, width=width, label="Learned Fair")
axes[1].bar(x_pos + 0.5 * width, true_loaded, width=width, label="True Loaded")
axes[1].bar(x_pos + 1.5 * width, learned_loaded, width=width, label="Learned Loaded")

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([str(f) for f in faces])
axes[1].set_xlabel("Die Face")
axes[1].set_ylabel("Emission Probability")
axes[1].set_title("Emission Distributions (True vs Learned)")
axes[1].legend()

plt.show()


