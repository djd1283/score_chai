# score_chai
A chess AI which plays chess by scoring board positions.

Contains Tensorflow code to play chess games, and then apply either a reward or punishment
to the model if it won or lost. Over time, the model learns to score winning board positions
as better than losing ones, and is more likely to end up in those positions.

## Description of components

ai.py - contains convolutional model for analyzing board positions

play.py - the model plays chess against itself and updates its play style

human_play.py - use the terminal to play chess against the AI based on its learned strategy!
