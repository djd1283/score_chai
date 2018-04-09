"""Test some functions here."""
import random
import numpy as np

print('Testing testing')

scores = np.random.uniform(low=0.0, high=1.0, size=10)

# Test n select feature
n_select = 5
max_indices = np.argpartition(scores, -n_select)[-n_select:]
choice_index = random.sample(list(max_indices), 1)[0]

print(scores)
print(max_indices)

max_scores = [scores[index] for index in max_indices]
print(max_scores)

print(choice_index)