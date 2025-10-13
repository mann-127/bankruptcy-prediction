# Company Bankruptcy Prediction Using Machine Learning 💼📉

A machine learning project that predicts company bankruptcy based on 95 financial indicators using advanced feature engineering, dimensionality reduction, clustering analysis, and gradient boosting classification.

## 📋 Project Overview

This project tackles the challenge of predicting company bankruptcy using a comprehensive dataset of financial metrics. The solution implements sophisticated data preprocessing, feature engineering, unsupervised learning for company characterization, and supervised classification to achieve accurate bankruptcy predictions.

## 🎯 Key Features

- **Advanced Feature Engineering**: Reduces 95 features to 40 principal components while retaining 95%+ information
- **Composite Feature Creation**: Combines highly correlated features (|ρ| ≥ 0.95) into meaningful composites
- **Normality Transformation**: Applies Yeo-Johnson power transformation to ensure feature normality
- **Company Clustering**: Uses K-Means clustering to identify company subgroups with similar characteristics
- **Gradient Boosting Classification**: Implements robust ensemble learning for bankruptcy prediction
- **Production-Ready Pipeline**: Includes train/test data processing and submission generation

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

- **Algorithm**: Gradient Boosting Classifier
- **Configuration**: 100 estimators with random_state=42
- **Training**: Fitted on PCA-transformed features
- **Output**: Binary bankruptcy predictions

## 🛠️ Technologies Used

- **Python 3.11+**
- **scikit-learn** - Machine learning algorithms and preprocessing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **SciPy** - Statistical functions and tests
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/mann-127/bankruptcy-prediction.git
cd bankruptcy-prediction
```

2. Open and run the notebooks:
```bash
jupyter notebook
```

3. Execute notebooks in sequence:
   - `model(1).ipynb` - Initial exploration
   - `model(2).ipynb` - Feature engineering experiments
   - `model(3).ipynb` - Clustering analysis
   - `model(4).ipynb` - Final model and predictions

4. The pipeline will:
   - Load and preprocess training data
   - Engineer composite features
   - Apply PCA transformation
   - Perform K-Means clustering
   - Train Gradient Boosting model
   - Generate predictions on test data
   - Create submission file

## 📁 Project Structure

```
bankruptcy-prediction/
├── model(1).ipynb        # Initial data exploration
├── model(2).ipynb        # Feature engineering experiments
├── model(3).ipynb        # Clustering analysis
├── model(4).ipynb        # Final model implementation
├── train_data.csv        # Training dataset (6.4 MB)
├── test_data.csv         # Test dataset (1.1 MB)
├── submission.csv        # Prediction output file
├── LICENSE               # MIT License
└── README.md             # Project documentation
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
GradientBoostingClassifier(n_estimators=100, random_state=42)
- Ensemble learning approach
- Robust to imbalanced data
- High predictive accuracy
```

## 📊 Model Performance

- **Dimensionality Reduction**: 95 → 40 features
- **Variance Retained**: 95.33%
- **Cluster Groups**: 3 distinct company profiles identified
- **Submission**: Predictions generated for test dataset

## 🔬 Advanced Features

1. **Correlation-Based Feature Extraction**: Automatically identifies and combines highly correlated features
2. **Statistical Validation**: Ensures all features pass normality tests
3. **Outlier Handling**: Uses robust scaling and data clipping (-10, 10 range)
4. **Consistent Preprocessing**: Same transformations applied to train and test data
5. **Reproducibility**: Fixed random seeds for consistent results

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest feature engineering improvements
- Optimize model hyperparameters
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**mann-127**
- GitHub: [@mann-127](https://github.com/mann-127)

## 🙏 Acknowledgments

- Dataset source: Financial bankruptcy indicators
- Feature engineering methodology based on correlation analysis
- Clustering techniques for financial profiling
- scikit-learn community for excellent ML tools

---

⭐ Star this repository if you find it helpful!
