import numpy as np
from hmmlearn.hmm import CategoricalHMM
import networkx as nx
import matplotlib.pyplot as plt

states = ["Difficult", "Medium", "Easy"]
observations = ["FB", "B", "S", "NS"]

start_probability = np.array([1/3, 1/3, 1/3], dtype=float)

transition_probability = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
], dtype=float)

emission_probability = np.array([
    [0.10, 0.20, 0.40, 0.30],
    [0.15, 0.25, 0.50, 0.10],
    [0.20, 0.30, 0.40, 0.10],
], dtype=float)

obs_seq_labels = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
obs_index = {label: i for i, label in enumerate(observations)}
obs_seq = np.array([obs_index[x] for x in obs_seq_labels], dtype=int).reshape(-1, 1)

model = CategoricalHMM(n_components=3)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

log_likelihood = model.score(obs_seq)
likelihood = float(np.exp(log_likelihood))

viterbi_logprob, viterbi_states = model.decode(obs_seq, algorithm="viterbi")
viterbi_prob = float(np.exp(viterbi_logprob))

print("Most likely hidden states (indices):", viterbi_states)
print("b) Log-likelihood:", log_likelihood)
print("   Likelihood    :", likelihood)
print("c) Viterbi logP :", viterbi_logprob)
print("   Viterbi P    :", viterbi_prob)

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
    plt.figure(figsize=(7,5))
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

# call it after defining your matrices:
draw_state_diagram(states, transition_probability)
