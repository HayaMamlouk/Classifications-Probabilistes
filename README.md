# Probabilistic Classifications Project

### Overview
This project explores various probabilistic classification methods within the context of a heart disease dataset. Using both basic and advanced probabilistic models, the project aims to predict the presence of heart disease based on patient attributes.

### Dataset
The heart disease dataset used for this project is derived from the Cleveland database and contains 14 primary attributes that are frequently referenced in machine learning research.

### Project Structure
- **`projet.py`**: Contains all custom code implementations, including class definitions, functions for probabilistic calculations, and classification models.
- **`utils.py`**: Utility functions and classes to support various tasks within the project, such as data discretization and visualization.
- **Data**: The project uses two pre-processed CSV files:
  - `train.csv`: Training data.
  - `test.csv`: Testing/validation data.

### Methods Implemented
1. **Prior Probability Classification**
   - Calculates prior probabilities and constructs an `APrioriClassifier` for a basic classification approach.
2. **2D Probabilistic Classifications**
   - Includes maximum likelihood (ML) and maximum a posteriori (MAP) classifiers that rely on joint probabilities.
3. **Naive Bayes Classifier**
   - Implements both ML and MAP versions of Naive Bayes, assuming attribute independence given the target.
4. **Complexity Evaluation**
   - Functions to compute memory complexity for conditional probability tables, under assumptions of complete and partial independence.
5. **Tree-Augmented Naive Bayes (TAN) Classifier**
   - Constructs a TAN classifier using conditional mutual information to determine optimal attribute connections.

### How to Run
1. Clone the repository.
2. Ensure `train.csv` and `test.csv` are available in the `data` directory.
3. Run the Jupyter notebook to execute the project steps and evaluate the classifiers.

### Results & Analysis
Classifiers are evaluated based on precision and recall metrics. Graphical representation of classifier performance across precision-recall space is provided.

### Conclusion
The project illustrates the strengths and limitations of various Bayesian classifiers, with conclusions drawn on their applicability to probabilistic classification tasks.
