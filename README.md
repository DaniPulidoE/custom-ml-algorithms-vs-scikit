# Fundamentals of Machine Learning - Course Projects - Autumn 2024

This repository shows my practical project work from the "Fundamentals of Machine Learning" course at Universidad Autónoma de Madrid (UAM) during the 2024/2025 academic year.

**Note:** To comply with university academic integrity policies, the source code for these projects is kept private. This README serves as a detailed summary and portfolio of the work completed.

## Technologies & Concepts

This project demonstrates practical experience with a various machine learning concepts and libraries, including:

* **Python Libraries:** `PySpark`, `scikit-learn`, `pandas`, `numpy`, `scipy`
* **Supervised Learning (Classification):**
    * Naive-Bayes (Gaussian, Laplace Correction)
    * K-Nearest Neighbors (K-NN)
    * Logistic Regression (with Gradient Descent)
* **Unsupervised Learning (Clustering):**
    * K-Means
* **Big Data Frameworks:**
    * **PySpark** (for distributed K-NN)
* **Metaheuristic Optimization:**
    * Genetic Algorithms for rule-based classification
* **Model Evaluation & Validation:**
    * ROC Analysis (ROC Space & ROC Curve)
    * Area Under the Curve (AUC)
    * Confusion Matrix
    * Cross-Validation & Simple Validation
* **Data Preprocessing:**
    * Data Standardization
    * Handling continuous vs. categorical data

---

## Project Breakdown

### Project 1: ML Algorithms from Scratch (Python, Scikit-learn & PySpark)

This project focused on implementing fundamental classification and clustering algorithms from scratch, and benchmarking their performance with respect to the optimized scikit-learn implementations.

* **Algorithms Implemented:** Developed **K-NN**, **K-Means**, and **Logistic Regression** algorithms from the ground up using only Python,  NumPy and Pandas.
* **Benchmarking:** Systematically benchmarked the performance of my custom algorithms against `scikit-learn`'s optimized versions to analyze differences in accuracy and efficiency.
* **Comparative Analysis:** Conducted a deep comparative study of Naive-Bayes, K-NN, and Logistic Regression using **ROC Analysis** (ROC Space, ROC Curve, and AUC) to evaluate classifier performance beyond simple accuracy.
* **Datasets:** `heart.csv`, `wdbc.csv`, `iris.csv`.

---

### Project 2: K-NN optimization with PySpark

This project focused on improving the execution speed of the custom K-NN algorithm by paralellizing its execution using PySpark, showcasing a direct application of distributed computing for machine learning.

* **Objective:** Improve the performance of the custom K-NN implementation, by distributing the distance calculations across a multi-thread and a multi-node PySpark cluster.
* **Key Objectives:**
    * Re-implemented the K-NN algorithm using PySpark's RDDs (Resilient Distributed Datasets).
    * Designed and deployed a PySpark cluster across multiple computers to run the algorithm in a true distributed environment.
* **Skills Gained:** Distributed computing, algorithm parallelization, and big data fundamentals.

---

### Project 3: Classification with Genetic Algorithms

This project consisted on implementing a Genetic Algorithm (GA) for a classification problem, training it to discover a set of `IF-THEN` classification rules.

* **Objective:** Designed and implemented a complete Genetic Algorithm from scratch to evolve a "rule base" capable of solving a classification problem.
* **Key Tasks:**
    * Designed a binary string representation for individuals, where each individual represented a complete set of classification rules.
    * Implemented core genetic operators, including **Crossover**, **Mutation**, **Roulette Wheel Selection**, and **Elitism**.
    * Defined a **fitness function** based on the classification accuracy of an individual's rule set on the training data.
    * Analyzed the GA's performance by varying parameters like population size, number of generations, and the maximum number of rules.
* **Datasets:** `titanic.csv`, `balloons.csv`.

---
