# Machine Learning Repository

This repository contains various Machine Learning implementations in Python, including algorithms for classification, regression, and data analysis.

## 📁 Repository Contents

- **`Decission_tree.py`** - Decision Tree classifier implementation with Google Colab integration
- **`Graph plotting.py`** - Data visualization and plotting utilities
- **`KNN`** - K-Nearest Neighbors algorithm implementation
- **`Multi-Regression.py`** - Multiple regression analysis implementation
- **`Naive_byes.py`** - Naive Bayes classifier implementation

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Usage
Each Python file can be run independently. For Google Colab files, upload them to your Colab environment.

## 🤖 GitHub Copilot Setup

Having trouble with GitHub Copilot subscription or VSCode integration? Check out our comprehensive setup guide:

**[📖 GitHub Copilot Setup and Troubleshooting Guide](./GITHUB_COPILOT_SETUP.md)**

This guide covers:
- How to check your current Copilot subscription
- How to subscribe to GitHub Copilot plans
- Common VSCode errors and solutions
- Optimal setup for ML development with Copilot

## 🧠 Machine Learning Algorithms Included

### 1. Decision Tree
- Binary classification implementation
- Feature importance analysis
- Accuracy evaluation with test/train split

### 2. K-Nearest Neighbors (KNN)
- Distance-based classification
- Customizable k-value selection

### 3. Multiple Regression
- Linear regression with multiple variables
- Statistical analysis and predictions

### 4. Naive Bayes
- Probabilistic classification
- Text and numerical data support

### 5. Data Visualization
- Graph plotting utilities
- Statistical visualization tools

## 📊 Example Usage

```python
# Example: Using Decision Tree classifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load your data
df = pd.read_csv('your_dataset.csv')

# Prepare features and target
X = df.drop(['target_column'], axis=1)
y = df['target_column']

# Train the model
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# Make predictions
predictions = dtree.predict(X_test)
```

## 🛠️ Development Setup

For the best development experience with this repository:

1. **Install Python dependencies**
2. **Set up GitHub Copilot** (see our guide above)
3. **Configure VSCode** with Python and Jupyter extensions
4. **Use virtual environment** for dependency management

## 📝 Notes

- Some files are configured for Google Colab (notice the `drive.mount()` calls)
- Modify file paths according to your local setup
- Consider using Jupyter notebooks for interactive development

## 🤝 Contributing

Feel free to contribute by:
- Adding new ML algorithms
- Improving existing implementations
- Adding documentation and examples
- Reporting issues or suggesting improvements

## 📞 Support

- For repository-specific questions: Create an issue
- For GitHub Copilot setup help: See [GITHUB_COPILOT_SETUP.md](./GITHUB_COPILOT_SETUP.md)
- For general ML questions: Check the code comments and documentation

---

**Happy Machine Learning! 🎯**