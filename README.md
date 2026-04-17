[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/sh8BuL_6)
# AI Stats Lab

## Objective

Let X ~ Uniform(0, a)

You must compute theoretical and sample-based statistics and analyze the transformation:

Z = 2X + 1

---

## Task

Implement:

def uniform_analysis(a, n_samples=10000)

in AI_stats_lab.py

---

## Return Format

Return:

(
    theoretical_mean,
    theoretical_variance,
    sample_mean,
    sample_variance,
    mean_error,
    variance_error,
    transformed_mean,
    transformed_variance
)

---

---

## Errors

mean_error = abs(sample_mean - theoretical_mean)

variance_error = abs(sample_variance - theoretical_variance)

---

## Requirements

Use:

- numpy.random.uniform
- numpy.mean
- numpy.var

Do NOT:

- print anything
- change return structure

---

## Run Tests

pytest test_AIstats_lab.py -q

---

## Setup (GitHub Classroom)

pip install numpy pytest

---
