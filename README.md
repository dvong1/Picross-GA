# Picross-GA
## Description
Picross, more commonly known as nonogram puzzles are NP-complete problems where optimal solutions are computationally expensive but easy/quick to verify. To find optimal solutions for NP-complete problems, methods such as brute-force, search algorithms and heuristics are applied to reduce search space in favor of more optimal solutions.   

This Final Project for CSE 545 Fall 2025 aims to utilize a genetic algorithm to solve any given nonogram puzzle. 



## What are Nonograms?
* Input image here of empty nonogram puzzle, and a solved nonogram puzzle




Group:
David Vong
Tyler
Seth

# Project Architecture
* /non is a dataset sub-directory that contains .non files which include clues, goal state, height and width of that puzzle
* data.py contains functions for data collection with the **pandas** library
* viz.py contains visuals for displaying the GA applied onto random nonogram solutions
* GA.py contains function and class defintions for components of a standard GA: TerminationCritera, population, chromosome, genes, genetic operators, etc.
* main.py will run the GA and display a live view of a random solution evolving to solve the puzzle.


# How to install
 1. After cloining the repo in your local environment, run the following code
`pip install -r requirements.txt`
