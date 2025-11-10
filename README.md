# Picross-GA
## Description
Picross, more commonly known as nonogram puzzles are NP-complete problems where optimal solutions are computationally expensive but easy/quick to verify. To find optimal solutions for NP-complete problems, methods such as brute-force, search algorithms and heuristics are applied to reduce search space in favor of more optimal solutions.   

This Final Project for CSE 545 Fall 2025 aims to utilize a genetic algorithm to solve any given nonogram puzzle. 
- Group members: David Vong, Tyler, Seth, Hieu Duong, Spencer, Robert

## What are Nonograms?
Nonograms - also known as Picross or Griddlers - are deceptively simple logic puzzles that quickly grow into formidable computational challenges as their size increases. A nonogram presents a two-dimensional grid with numerical clues that indicate how many consecutive cells in each row and column should be filled. An example of an empty and solved nonogram is shown in the figure below, illustrating how numerical clues define the sequences of filled cells along each row and column.


![An example of an empty and solved nonogram of letter "P"](nonogram1.jpg)



# Features
- Multi-format input: Supports both .non and .json Nonogram puzzle files (Nonogram should be exported from: https://nonogram.kniffen.dev/).
- Wisdom-of-Crowds recombination: Instead of two-parent crossover, multiple elite individuals contribute via majority voting, encouraging stable convergence.
- Adaptive mutation operators: Random bit flips for exploration; Row and column sum-matching to maintain valid clue totals.
- Elitist selection and replacement: Ensures the best individuals persist between generations.
- Visualization:
  - Trend plots of best fitness over generations.
  - Side-by-side goal vs. outcome comparisons.
  - (Optional) animated GIFs showing grid evolution



# How to install
 1. Requirement: Python >= 3.10
 2. After cloning the repo in your local environment, run the following code to install necessary libraries:
`pip install -r requirements.txt`



# Execution
- To batch-run multiple Nonogram puzzles with different configurations, use the included shell script:
`bash non_runs_2_biggening.sh`
- To reproduce the results in the report, run:
`bash example_run.sh`
Each run logs results, saves visualizations, and outputs runtime statistics automatically to non_ga_results/.
- Example single-run execution with spade.non:
`python3 nonogram_woc_ga.py --non nons/spade.non --pop 200 --gens 400 --elite 0.1 --crowd_ratio 0.2 --tsize 4 --m_parents 4 --trendplot --comparepng`


