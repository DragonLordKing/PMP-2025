from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

m = DiscreteBayesianNetwork([("O","H"), ("O","W"), ("H","R"), ("W","R"), ("H","E"), ("R","C")])

print("Local(O):", m.local_independencies("O"))
print("Local(L):", m.local_independencies("H"))
print("Local(M):", m.local_independencies("W"))
print("Local(O):", m.local_independencies("R"))
print("Local(L):", m.local_independencies("E"))
print("Local(M):", m.local_independencies("C"))

print("\n", m.get_independencies())

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
cpd_O = TabularCPD(
    variable="O",
    variable_card=2,
    values=[[0.3],        # O = cold
            [0.7]]        # O = mild
)

cpd_H = TabularCPD(
    variable="H",
    variable_card=2,
    values=[[0.9, 0.8],   # H = yes
            [0.1, 0.2]],  # H = no
    evidence=["O"],
    evidence_card=[2]
)

cpd_W = TabularCPD(
    variable="W",
    variable_card=2,
    values=[[0.1, 0.4],   # W = yes
            [0.9, 0.6]],  # W = no
    evidence=["O"],
    evidence_card=[2]
)

cpd_R = TabularCPD(
    variable="R",
    variable_card=2,
    values=[[0.5, 0.7, 0.1, 0.4],  # R = cool
            [0.5, 0.3, 0.9, 0.6]], # R = warm
    evidence=["H", "W"],
    evidence_card=[2, 2]
)

cpd_E = TabularCPD(
    variable="E",
    variable_card=2,
    values=[[0.8, 0.8],   # E = high use
            [0.2, 0.2]],  # E = low use
    evidence=["H"],
    evidence_card=[2]
)

cpd_C = TabularCPD(
    variable="C",
    variable_card=2,
    values=[[0.85, 0.6],  # C = comfortable
            [0.15, 0.4]], # C = uncomfortable
    evidence=["R"],
    evidence_card=[2]
)

model = DiscreteBayesianNetwork([
    ("O", "H"),
    ("O", "W"),
    ("H", "R"),
    ("W", "R"),
    ("H", "E"),
    ("R", "C")
])

model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)
model.check_model()


infer = VariableElimination(model)

query_result = infer.query(
    variables=["H"],
    evidence={"C": 0}
)
query_result2 = infer.query(
    variables=["E"],
    evidence={"C": 0}
)

print(query_result)
print(query_result2)
print("P(H = yes | C = comfortable) =", float(query_result.values[1]))
print("P(HE = high | C = comfortable) =", float(query_result2.values[1]))

from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

map_hw = infer.map_query(
    variables=["H", "W"],
    evidence={"C": 0},
    show_progress=False
)

print(map_hw)


#2
import numpy as np
from hmmlearn.hmm import CategoricalHMM
import networkx as nx
import matplotlib.pyplot as plt

states = ["Walking", "Running", "Resting"]
observations = ["Low", "Medium", "High"]

start_probability = np.array([1/3, 1/3, 1/3], dtype=float)

transition_probability = np.array([
    [0.6, 0.3, 0.1],  # from Walking
    [0.2, 0.7, 0.1],  # from Running
    [0.3, 0.2, 0.5]   # from Resting
], dtype=float)

emission_probability = np.array([
    [0.1, 0.7, 0.2],   # Walking  -> Low, Medium, High
    [0.05, 0.25, 0.7], # Running  -> Low, Medium, High
    [0.8, 0.15, 0.05], # Resting  -> Low, Medium, High
], dtype=float)

obs_seq_labels = ["Low", "Medium", "High", "High", "Medium","Low", "Low", "Medium", "High", "Medium", "Low"]

obs_index = {label: i for i, label in enumerate(observations)}
obs_seq = np.array([obs_index[x] for x in obs_seq_labels], dtype=int).reshape(-1, 1)

model = CategoricalHMM(n_components=len(states))
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

log_likelihood = model.score(obs_seq)
likelihood = float(np.exp(log_likelihood))

viterbi_logprob, viterbi_states = model.decode(obs_seq, algorithm="viterbi")
viterbi_prob = float(np.exp(viterbi_logprob))

print("=== hmmlearn results ===")
print("Most likely hidden states (indices):", viterbi_states)
print("Most likely hidden states (names):  ",
      [states[i] for i in viterbi_states])
print("Log-likelihood (score):", log_likelihood)
print("Likelihood           :", likelihood)
print("Viterbi logP         :", viterbi_logprob)
print("Viterbi P            :", viterbi_prob)

def forward_algorithm(pi, A, B, obs):
    N = A.shape[0]
    T = len(obs)
    alpha = np.zeros((T, N))

    alpha[0, :] = pi * B[:, obs[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, obs[t]] * np.dot(alpha[t - 1, :], A[:, j])

    seq_prob = np.sum(alpha[-1, :])
    return alpha, seq_prob

obs_flat = obs_seq.flatten()
alpha, forward_prob = forward_algorithm(start_probability, transition_probability, emission_probability, obs_flat)

print("=== Manual Forward Algorithm ===")
print("Forward probability P(O_0..O_T-1):", forward_prob)
print("Forward log-probability         :", np.log(forward_prob))

def viterbi_algorithm(pi, A, B, obs):
    N = A.shape[0]
    T = len(obs)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0, :] = pi * B[:, obs[0]]
    psi[0, :] = 0

    for t in range(1, T):
        for j in range(N):
            temp = delta[t - 1, :] * A[:, j]
            psi[t, j] = np.argmax(temp)
            delta[t, j] = B[j, obs[t]] * np.max(temp)

    best_last_state = np.argmax(delta[T - 1, :])
    best_path_prob = delta[T - 1, best_last_state]

    best_path = [0] * T
    best_path[T - 1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return best_path_prob, best_path

vit_prob_manual, vit_path_manual = viterbi_algorithm(start_probability, transition_probability, emission_probability, obs_flat)

rng = np.random.default_rng(0)

def sample_hmm(pi, A, B, T, rng):
    N = A.shape[0]
    state = rng.choice(N, p=pi)
    obs = np.zeros(T, dtype=int)
    states_path = np.zeros(T, dtype=int)
    obs[0] = rng.choice(B.shape[1], p=B[state])
    states_path[0] = state
    for t in range(1, T):
        state = rng.choice(N, p=A[state])
        obs[t] = rng.choice(B.shape[1], p=B[state])
        states_path[t] = state
    return obs, states_path

n_sequences = 10000
matches = 0
T = len(obs_flat)

for _ in range(n_sequences):
    sample_obs, _ = sample_hmm(start_probability, transition_probability, emission_probability, T, rng)
    if np.array_equal(sample_obs, obs_flat):
        matches += 1

empirical_prob = matches / n_sequences

print("Forward probability:", forward_prob)
print("Forward log-probability:", np.log(forward_prob))
print("Empirical probability (10000 samples):", empirical_prob)
print("Difference (empirical - forward):", empirical_prob - forward_prob)

print("Best path probability:", vit_prob_manual)
print("Best path (indices):", vit_path_manual)
print("Best path (names):", [states[i] for i in vit_path_manual])
print("Best path log-prob:", np.log(vit_prob_manual))
print()

print("hmmlearn log-likelihood vs forward log-likelihood:")
print("hmmlearn:", log_likelihood)
print("forward :", np.log(forward_prob))

print("hmmlearn Viterbi logP vs manual Viterbi logP:")
print("hmmlearn:", viterbi_logprob)
print("manual:", np.log(vit_prob_manual))

print("Paths equal?", np.all(viterbi_states == np.array(vit_path_manual)))

def draw_state_diagram(state_names, A):
    G = nx.DiGraph()
    for i, s in enumerate(state_names):
        G.add_node(i, label=s)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            p = A[i, j]
            if p > 0:
                G.add_edge(i, j, label=f"{p:.2f}", weight=p)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="#e6f0ff",
                           edgecolors="#1b4f72", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={i: state_names[i] for i in range(len(state_names))})

    widths = [2 + 6 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=15, width=widths)

    edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("HMM State Diagram (Transition Probabilities)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

draw_state_diagram(states, transition_probability)