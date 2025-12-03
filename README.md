# 🔍 Bias Audit: Machine Learning Fairness in Income Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> A comprehensive bias audit of machine learning models trained on the Adult Income dataset, analyzing fairness across gender, race, and age demographics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Deliverables](#deliverables)
- [Technologies Used](#technologies-used)
- [References](#references)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Overview

This project conducts a thorough **fairness audit** of machine learning models predicting income levels (>50K vs ≤50K) using the UCI Adult Income dataset. We identify algorithmic bias across protected demographic groups and implement multiple mitigation strategies to improve fairness while maintaining model performance.

### **Research Questions**
- How do ML models exhibit bias across gender, race, and age?
- What fairness metrics best capture discriminatory patterns?
- Can bias mitigation techniques reduce discrimination without significant accuracy loss?
- What are the real-world implications of algorithmic bias in income prediction?

### **Protected Attributes Analyzed**
- 👥 **Gender** (Male vs Female)
- 🌍 **Race** (White vs Non-White)
- 📅 **Age** (≥25 vs <25)

---

## 🚨 Key Findings

### Baseline Model Bias (Before Mitigation)

| Protected Attribute | Statistical Parity Difference | Disparate Impact | Status |
|---------------------|-------------------------------|------------------|--------|
| **Gender** | -0.196 | 0.563 | ⚠️ Severe Bias |
| **Race** | -0.252 | 0.438 | ⚠️ Severe Bias |
| **Age** | -0.331 | 0.311 | ⚠️ Critical Bias |

### After Mitigation (Reweighing Applied)

| Protected Attribute | SPD Improvement | DI Improvement | Overall Improvement |
|---------------------|-----------------|----------------|---------------------|
| **Gender** | -0.196 → -0.098 | 0.563 → 0.751 | **50% reduction in bias** |
| **Race** | -0.252 → -0.142 | 0.438 → 0.632 | **44% reduction in bias** |
| **Age** | -0.331 → -0.186 | 0.311 → 0.489 | **57% reduction in bias** |

**📊 Performance Trade-off:** Accuracy decreased from 84.7% to 83.1% (1.6% loss) — an acceptable trade-off for significant fairness improvements.

---

## 📁 Project Structure

```
bias-audit-project/
│
├── notebooks/
│   └── bias_audit_analysis.ipynb          # Main analysis notebook (Sections 1-7)
│
├── data/
│   └── README.md                          # Data source information
│
├── presentation/
│   ├── bias_audit_slides.html             # Interactive HTML presentation
│   └── slides_content.md                  # PowerPoint content guide
│
├── reports/
│   ├── ethics_statement.md                # 500-word ethics statement
│   └── reference_list.md                  # Comprehensive bibliography
│
├── visualizations/
│   ├── fairness_metrics_baseline.png      # Baseline bias visualizations
│   ├── mitigation_results.png             # Before/after comparisons
│   └── intersectional_analysis.png        # Intersectional bias patterns
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── LICENSE                                # MIT License

```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/ZezeNteyi99/Bias_Audit.git
cd Bias_Audit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/bias_audit_analysis.ipynb
```

### Required Libraries
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
aif360>=0.5.0
fairlearn>=0.7.0
```

---

## 🚀 Usage

### Running the Complete Analysis

1. **Open the Jupyter notebook** in Google Colab or locally
2. **Run all cells sequentially** (Sections 1-7)
3. **View generated visualizations** in the notebook output
4. **Export results** to the `visualizations/` folder

### Quick Start (Google Colab)

```python
# Section 1: Install libraries
!pip install aif360 fairlearn scikit-learn pandas numpy matplotlib seaborn

# Section 2: Load data
import pandas as pd
train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(train_url, names=column_names, sep=',\s*', na_values=['?'], engine='python')

# Run remaining sections...
```

### Reproducing Results

All results are **fully reproducible** using the provided notebook with `random_state=42` for consistency.

---

## 🔬 Methodology

### 1. **Data Preprocessing**
- Loaded Adult Income dataset (45,000+ records)
- Created binary protected attributes (Gender, Race, Age)
- Encoded categorical features using LabelEncoder
- Split data: 70% training, 30% testing

### 2. **Baseline Model Training**
- Algorithm: Logistic Regression
- Features: 13 demographic and economic variables
- Performance: 84.7% accuracy

### 3. **Fairness Metrics Calculated**
- **Statistical Parity Difference (SPD)** - Measures equal positive outcome rates
- **Disparate Impact (DI)** - Ratio of positive outcomes between groups
- **Equal Opportunity Difference (EOD)** - Measures equal true positive rates
- **Average Odds Difference (AOD)** - Average of TPR and FPR differences

### 4. **Bias Mitigation Techniques**
- **Reweighing** (Pre-processing) - Adjusts training sample weights
- **Fair Classifier** (In-processing) - Adds fairness regularization
- **Threshold Optimization** (Post-processing) - Group-specific decision boundaries

### 5. **Intersectional Analysis**
- Analyzed bias at intersections (e.g., Female + Non-White)
- Identified compounded discrimination patterns

---

## 📊 Results

### Mitigation Effectiveness

```
✅ Gender Bias:  50% reduction (SPD: -0.196 → -0.098)
✅ Race Bias:    44% reduction (SPD: -0.252 → -0.142)
✅ Age Bias:     57% reduction (SPD: -0.331 → -0.186)
```

### Key Insights

1. **Bias is Real:** Baseline model showed severe discrimination across all protected attributes
2. **Mitigation Works:** Reweighing reduced bias by 40-70% with minimal accuracy loss
3. **Intersectionality Matters:** Young, non-white women faced the most severe discrimination
4. **Trade-offs are Acceptable:** 1.6% accuracy loss is justified for significant fairness gains

### Real-World Implications

**Identified Harms:**
- 💳 Financial services: Biased credit scoring and lending
- 💼 Employment: Discriminatory resume screening and salary predictions
- 🏠 Housing: Unfair rent/mortgage approval algorithms
- ⚖️ Systemic inequality: Perpetuation of historical discrimination

---

## 📦 Deliverables

### 1. **Jupyter Notebook** 
Complete analysis with 7 sections covering data loading, EDA, model training, fairness metrics, bias mitigation, and comprehensive results.

### 2. **Presentation (7 slides)**
- Project overview
- Key bias patterns discovered
- Mitigation results
- Real-world implications
- Recommendations for stakeholders

### 3. **Ethics Statement (500 words)**
Connects technical findings to AI ethics principles including:
- Consequentialism, deontology, virtue ethics
- Fairness, accountability, transparency (FAT)
- Stakeholder responsibilities
- Ongoing vigilance requirements

### 4. **Reference List**
32 scholarly sources covering:
- Fairness definitions and metrics
- Bias mitigation techniques
- AI ethics frameworks
- Real-world case studies

---

## 💻 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Primary programming language |
| **Jupyter Notebook** | Interactive development environment |
| **Pandas & NumPy** | Data manipulation and numerical computing |
| **Scikit-learn** | Machine learning models and metrics |
| **AIF360** | IBM AI Fairness toolkit for bias detection |
| **Fairlearn** | Microsoft fairness assessment toolkit |
| **Matplotlib & Seaborn** | Data visualization |
| **Google Colab** | Cloud-based notebook execution |

---

## 📚 References

Key papers and resources used in this project:

- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*
- Mehrabi, N., et al. (2021). Survey on Bias and Fairness in Machine Learning. *ACM Computing Surveys*
- Bellamy, R. K. E., et al. (2019). AI Fairness 360. *IBM Journal of Research and Development*
- Buolamwini, J., & Gebru, T. (2018). Gender Shades: Intersectional Accuracy Disparities
- UCI Machine Learning Repository: Adult Dataset

[**Full reference list →**](reports/reference_list.md)

---

## 👥 Contributors

- **Khanyisa Zezethu Nteyi** - Data Scientist & Project Lead
- **Thembani Nkuna** - Technical Implementation
- **Sphumelele Ngobese** - Visualization Specialist
- **Kazimla Nkabalaza** - Fairness Analysis
- **Fikile Noyila** - Ethics & Documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Adult Income dataset
- **IBM Research** for the AIF360 toolkit
- **Microsoft Research** for the Fairlearn library
- **Course Instructors** for guidance on ethical AI practices

---

## 📞 Contact

**Project Maintainer:** Khanyisa Zezethu Nteyi
- 📧 Email: zezenteyi99@gmail.com
- 💼 LinkedIn: [linkedin.com/in/khanyisa-zezethu-nteyi-765390207](https://linkedin.com/in/khanyisa-zezethu-nteyi-765390207)
- 🐙 GitHub: [@ZezeNteyi99](https://github.com/ZezeNteyi99)

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! It helps others discover this work.

---

<p align="center">
  <i>Built with ❤️ for algorithmic fairness and social justice</i>
</p>

<p align="center">
  <sub>Last Updated: December 2025</sub>
</p>
