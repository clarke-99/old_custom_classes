# Old Custom Classes

## Overview
This repository contains a collection of custom classes that were used in my earlier machine-learning projects. These classes were instrumental in handling various tasks and extending the functionality of standard libraries. However, they were developed without the use of Git and before I fully understood the importance of modularity and efficient coding practices.

## Background
The classes in this repository, while functional, are computationally expensive and challenging to modify. The lack of modular design makes it difficult to update or extend these classes without risking loss of functionality. As a result, I have started remaking these classes with improved coding practices to reduce computational costs and enhance ease of maintenance.

## Classes Included
- **Class 1: EDA_V1**
  - **Purpose:** Designed to streamline the preprocessing and exploratory data analysis (EDA) of datasets.
  - **Key Features:** 
    - Displays summary statistics.
    - Encodes categorical data.
    - Determines normality of distributions.
    - Removes outliers using parametric and non-parametric methods.
    - Creates correlation heatmaps.
    - Calculates and graphically displays the Variance Inflation Factor (VIF) for each feature.
    - Analyses feature distributions and creates QQ plots.
    - Transforms data based on a user-defined skew threshold, then compares the transformations using various statistical tests to assess normality, recreating QQ plots for those that pass.
    - Automates the identification of epsilon and theta parameters for DBSCAN.
    - Automatically saves graphs generated during the process using the project name.

- **Class 2: Model Builder**
  - **Purpose:** Facilitates the rapid creation, training, and tuning of various classifier algorithms.
  - **Key Features:**
    - Removes features using Recursive Feature Elimination (RFE).
    - Assesses feature importance using permutation feature importance.
    - Generates a variety of performance metrics.
    - Automatically saves graphs generated during the process using the project name.

## Lessons Learned
Creating these classes without version control (Git) made the development process time-consuming, inefficient, and prone to errors. These experiences highlighted several key points:

- **Modularity:** Designing code in small, independent modules simplifies testing, debugging, and extending functionality.
- **Efficiency:** Writing code optimised for performance is crucial, especially for computationally intensive tasks in machine learning.
- **Version Control:** Using Git or other version control systems to track changes, collaborate more effectively, and maintain a history of the codebase is essential.
- **Code Organization:** Properly organizing code and determining the sequence of actions is critical. For example:
  - **Order of Operations:** Ensuring that tasks such as determining linearity, checking for normality, and removing outliers are performed in the correct order can greatly impact the effectiveness of preprocessing.
  - **Function Placement:** Placing related functions within cohesive modules and clearly defining their responsibilities can prevent code bloat and make the overall structure more maintainable.

## Future Work
I am currently in the process of remaking these classes with a focus on:
- Improved modularity to allow for easier modifications and extensions.
- Optimised algorithms to reduce computational expense.
- Better documentation to make the classes more user-friendly and maintainable.

## Contributing
While this repository serves as a historical reference to my earlier work, contributions are welcome if you have suggestions for improving or refactoring these classes. 
