Image Feature Extraction and ClassificationThis repository contains an end-to-end Python pipeline designed to extract features from image datasets and classify them using machine learning. 

The workflow supports both traditional feature extraction (Local Binary Patterns - LBP) and deep learning feature extraction (VGG16).

⚙️ FeaturesAutomated Dataset Unzipping: Safely extracts your .zip dataset files directly in your environment.

Image Preprocessing: Reads and resizes images into arrays for standardized processing.

Feature Extraction:LBP (Local Binary Patterns): Extracts texture-based feature vectors.

VGG16 (CNN): Extracts deep, robust bottleneck features from the fc2 layer.

Model Training: Splits your dataset and trains either a Support Vector Machine (SVM) or Logistic Regression classifier.

Model Evaluation: Generates a confusion matrix (visualized with Seaborn) and a classification report.


🚀 Getting StartedPrerequisitesEnsure you have the following core Python libraries installed in your environment. 

You can install them via pip:bashpip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow


UsageDataset Preparation: 

Place your dataset zip file (e.g., AT&T.zip) in the appropriate directory (e.g., /content/).

Execute Pipeline: Run the Python script to sequentially extract files, extract features, train the model, and print performance metrics.


📁 Code OverviewThe script handles the machine learning lifecycle in four sequential steps:

Extraction: Unpacks compressed dataset files for processing.

Feature Extraction: Iterates through image classes to map, resize, and pull feature vectors.

Training: Uses train_test_split (default 33% testing size) to train your selected classifier.

Evaluation: Outputs a heatmap for the confusion matrix and prints precision, recall, and F1-scores.
