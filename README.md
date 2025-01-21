# Graph-Based-Prediction-of-Protein-Stability-Changes-Induced-by-Single-Point-Mutations.
it is a tool which personally made by me
Background:
Protein stability, crucial for biological function, can be significantly affected by single-point mutations, altering Gibbs free energy. Accurate prediction methods include structure-based approaches, limited by the 0.5% availability of structural data in UniProt, and more broadly applicable but less accurate sequence-based methods. This study introduces a graph-based machine learning approach, combining structural and sequence data, to predict mutation effects on protein stability. Our method enhances prediction accuracy, even with limited structural data, improving insights into protein stability and genetic variations.
Methods:
We constructed graphs for wild-type and mutant proteins, using 9 physicochemical properties as node features. Our graph-based approach predicted Gibbs free energy changes. The model was validated using k-fold cross-validation to ensure robustness and prediction accuracy.
Results:
Our graph-based model demonstrated robust performance in predicting Gibbs free energy
changes, with mean squared error (MSE) and R² scores comparable to existing tools. This led to the development of a new tool that offers enhanced accuracy and reliability in predicting protein stability changes due to mutations.

