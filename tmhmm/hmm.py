import numpy as np


def viterbi(sequence, initial, transitions, emissions, char_map, label_map, name_map):
    sequence = sequence.upper()
    observations = [char_map[character] for character in sequence]
    state_count = len(initial)
    score_table = np.zeros((2, state_count), dtype=float)
    traceback = np.zeros((len(observations), state_count), dtype=np.intp)
    with np.errstate(divide="ignore"):
        score_table[0] = np.log(initial) + np.log(emissions[:, observations[0]])
        log_transitions = np.log(transitions)
        log_emissions = np.log(emissions)
    state_indices = np.arange(state_count)

    for position, observation in enumerate(observations[1:], start=1):
        previous_scores = score_table[(position - 1) % 2]
        candidates = previous_scores[:, None] + log_transitions
        previous_states = np.argmax(candidates, axis=0)
        traceback[position] = previous_states
        score_table[position % 2] = candidates[previous_states, state_indices] + log_emissions[:, observation]

    state = int(np.argmax(score_table[(len(observations) - 1) % 2]))
    path = []
    for position in range(len(observations) - 1, -1, -1):
        path.append(label_map[state])
        if position:
            state = int(traceback[position, state])
    return score_table, "".join(reversed(path))


def forward(sequence, initial, transitions, emissions, char_map, label_map, name_map):
    sequence = sequence.upper()
    observations = [char_map[character] for character in sequence]
    table = np.zeros((len(observations), len(initial)), dtype=float)
    constants = np.zeros(len(observations), dtype=float)
    table[0] = initial * emissions[:, observations[0]]
    constants[0] = table[0].sum()
    table[0] /= constants[0]

    for position, observation in enumerate(observations[1:], start=1):
        table[position] = (table[position - 1] @ transitions) * emissions[:, observation]
        constants[position] = table[position].sum()
        table[position] /= constants[position]
    return table, constants


def backward(sequence, constants, initial, transitions, emissions, char_map, label_map, name_map):
    sequence = sequence.upper()
    observations = [char_map[character] for character in sequence]
    table = np.zeros((len(observations), len(initial)), dtype=float)
    table[-1] = 1.0 / constants[-1]

    for position in range(len(observations) - 2, -1, -1):
        table[position] = transitions @ (
            table[position + 1] * emissions[:, observations[position]]
        )
        table[position] /= constants[position]
    return table