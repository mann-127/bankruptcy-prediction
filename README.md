# Company Bankruptcy Prediction Using Machine Learning 💼📉

A machine learning project that predicts company bankruptcy based on 95 financial indicators using advanced feature engineering, dimensionality reduction, clustering analysis, and ensemble learning with SMOTE for class imbalance handling.

## 📋 Project Overview

This project tackles the challenge of predicting company bankruptcy using a comprehensive dataset of financial metrics. The solution implements sophisticated data preprocessing, feature engineering, unsupervised learning for company characterization, and supervised classification to achieve accurate bankruptcy predictions with a focus on maximizing recall (catching actual bankruptcies).

## 🎯 Key Features

- **Advanced Feature Engineering**: Reduces 95 features to 40 principal components while retaining 95%+ information
- **Composite Feature Creation**: Combines highly correlated features (|ρ| ≥ 0.95) into meaningful composites
- **Normality Transformation**: Applies Yeo-Johnson power transformation to ensure feature normality
- **Company Clustering**: Uses K-Means clustering (k=3) to identify company subgroups with similar characteristics
- **SMOTE for Class Imbalance**: Implements Synthetic Minority Over-sampling Technique to handle imbalanced data
- **Multiple Ensemble Models**: Compares AdaBoost, Gradient Boosting, Random Forest, and other classifiers
- **Stratified K-Fold Cross-Validation**: 5-fold CV for robust model evaluation
- **Production-Ready Pipeline**: Complete preprocessing and model pipeline with persistence

## 📊 Dataset

The project uses financial data with:

### Input Features (95 original features):
- **Profitability Metrics**: ROA, Operating Margin, Net Profit Rate, etc.
- **Liquidity Ratios**: Current Ratio, Quick Ratio, Cash Flow metrics
- **Leverage Ratios**: Debt ratios, Liability metrics, Net Worth indicators
- **Efficiency Metrics**: Asset Turnover, Inventory Turnover, Receivable Turnover
- **Growth Indicators**: Revenue Growth, Asset Growth, Profit Growth
- **Per-Share Metrics**: EPS, Cash Flow per Share, Net Value per Share

### Target Variable:
- **Bankrupt?**: Binary indicator (0 = Solvent, 1 = Bankrupt)

## 🏗️ Project Methodology

### 1. Training Data Preparation

#### Feature Engineering & Reduction
- **Composite Features Created**:
  - `Liability_Composite`: Combined current liability metrics
  - `Net_Value_Per_Share_BC`: Merged net value per share variants
  - `Interest_Rate_Composite`: Combined pre/post-tax interest rates
  - `Gross_Margin_Composite`: Merged margin metrics
  - `EPS_Net_Profit_Composite`: Combined earnings metrics

#### Preprocessing Pipeline
1. **Robust Scaling**: Minimizes outlier impact using RobustScaler
2. **Feature Correlation Analysis**: Identifies |ρ| ≥ 0.95 correlations
3. **Normality Testing**: Statistical validation using normaltest
4. **Power Transformation**: Yeo-Johnson transformation for non-normal features
5. **Dimensionality Reduction**: PCA to 40 components (95.3% variance retained)

### 2. Company Characterization

#### Unsupervised Clustering
- **Algorithm**: K-Means clustering (k=3)
- **Purpose**: Group companies with similar financial profiles
- **Visualization**: 2D scatter plot using first two principal components
- **Output**: Cluster IDs for each company

### 3. Classification Model

- **Algorithms Evaluated**: 
  - AdaBoost Classifier (200 estimators, SAMME algorithm)
  - Gradient Boosting Classifier (100 estimators)
  - Random Forest Classifier (200 estimators)
  - Support Vector Machines (RBF kernel)
  - K-Nearest Neighbors
  - Optional: XGBoost and LightGBM
- **Validation**: 5-fold Stratified Cross-Validation
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Optimization Goal**: Maximize recall to minimize false negatives (missed bankruptcies)
- **Output**: Binary bankruptcy predictions with probability scores

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms and preprocessing
- **imbalanced-learn** - SMOTE for handling class imbalance
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **SciPy** - Statistical functions and tests
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development
- **Joblib** - Model persistence

### Optional Dependencies
- **XGBoost** - Extreme Gradient Boosting (optional)
- **LightGBM** - Light Gradient Boosting Machine (optional)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mann-127/bankruptcy-prediction.git
cd bankruptcy-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `bankruptcy_prediction_final.ipynb`

3. Select the Python kernel from your virtual environment

4. Run all cells to:
   - Load and preprocess training data
   - Engineer composite features
   - Apply PCA transformation
   - Perform K-Means clustering
   - Train and compare multiple models with cross-validation
   - Generate predictions on test data
   - Create submission file

### Expected Outputs

The notebook will generate:
- `preprocessor.pkl` - Fitted preprocessing pipeline
- `best_model_pipeline.pkl` - Best performing model
- `adaboost_pipeline.pkl` - AdaBoost model
- `gradientboosting_pipeline.pkl` - Gradient Boosting model  
- `kmeans_clustering.pkl` - Clustering model
- `submission_final.csv` - Final predictions for submission

## 📁 Project Structure

```
bankruptcy-prediction/
├── bankruptcy_prediction_final.ipynb  # Main notebook with complete pipeline
├── train_data.csv                     # Training dataset (6.4 MB)
├── test_data.csv                      # Test dataset (1.1 MB)
├── submission_final.csv               # Final prediction output
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
├── README.md                          # Project documentation
│
├── Model Artifacts (generated by notebook):
│   ├── preprocessor.pkl               # Fitted preprocessing pipeline
│   ├── best_model_pipeline.pkl        # Best performing model
│   ├── adaboost_pipeline.pkl          # AdaBoost classifier
│   ├── gradientboosting_pipeline.pkl  # Gradient Boosting classifier
│   └── kmeans_clustering.pkl          # K-Means clustering model
│
└── .venv/                             # Virtual environment (not tracked)
```

## 🔍 Key Implementation Details

### Feature Engineering Constraints
- Maximum 50 features allowed after reduction
- PCA only applied after feature engineering
- Original features kept only if non-linearity confirmed
- Composite features created from |ρ| ≥ 0.95 correlations
- Normality must be confirmed for all features

### Data Preprocessing
```python
# Scaling
RobustScaler() - Handles outliers effectively

# Normalization
PowerTransformer(method='yeo-johnson') - Ensures normality

# Dimensionality Reduction
PCA(n_components=0.95) - Retains 95% variance
Result: 40 components from 95 features
```

### Clustering Analysis
```python
K-Means(n_clusters=3, random_state=42)
- Identifies company financial profiles
- Visualized using PC1 vs PC2
- Centroids highlighted for interpretation
```

### Classification
```python
# Multiple models evaluated with SMOTE
AdaBoostClassifier(n_estimators=200, algorithm='SAMME')
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
RandomForestClassifier(n_estimators=200)
# ... and more

# Pipeline with SMOTE
make_pipeline(SMOTE(random_state=42), model)

# 5-fold Stratified Cross-Validation
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## 📊 Model Performance

- **Dimensionality Reduction**: 95 → 40 principal components (95.33% variance retained)
- **Cluster Groups**: 3 distinct company financial profiles identified
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Class Imbalance Handling**: SMOTE oversampling
- **Model Selection**: Best model chosen based on recall (minimizing missed bankruptcies)
- **Output**: Predictions with probability scores for test dataset

### Key Metrics Focus
- **Recall (Sensitivity)**: Prioritized to catch actual bankruptcies
- **Precision**: Balance between accuracy and recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs actuals

## 🔬 Advanced Features

1. **Correlation-Based Feature Extraction**: Automatically identifies and combines highly correlated features
2. **Statistical Validation**: Ensures all features pass normality tests after transformation
3. **Outlier Handling**: Uses robust scaling and data clipping (-10, 10 range)
4. **Consistent Preprocessing**: Identical transformations applied to train and test data
5. **SMOTE Integration**: Handles class imbalance during cross-validation
6. **Model Comparison**: Systematic evaluation of multiple algorithms
7. **Pipeline Persistence**: All transformers and models saved for reproducibility
8. **Detailed Metrics**: Comprehensive evaluation including recall, precision, F1, confusion matrix
9. **Reproducibility**: Fixed random seeds (RANDOM_STATE=42) for consistent results

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest feature engineering improvements
- Optimize model hyperparameters
- Add additional classifiers
- Improve documentation
- Submit pull requests

Please ensure your code follows the project structure and includes appropriate documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**mann-127**
- GitHub: [@mann-127](https://github.com/mann-127)

## 🙏 Acknowledgments

- Dataset source: Company financial bankruptcy indicators
- Feature engineering methodology based on statistical correlation analysis
- K-Means clustering for financial profiling
- SMOTE technique for handling imbalanced datasets
- scikit-learn and imbalanced-learn communities for excellent ML tools

---

**Note**: This project prioritizes recall over other metrics to minimize the risk of missing actual bankruptcies (false negatives), which is critical in bankruptcy prediction applications.

⭐ Star this repository if you find it helpful!
