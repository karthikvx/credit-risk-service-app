# Credit Risk Service App

## Overview

This repository contains a comprehensive Credit Risk Service App designed for real-time credit risk scoring, portfolio analysis, stress testing, and regulatory reporting. The system integrates machine learning models for Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD), along with AWS services for real-time data streaming and processing.

## Features

- **Real-time Risk Scoring**: Calculate PD, LGD, EAD, and expected loss for individual credit applications.
- **Portfolio Analysis**: Assess portfolio-level risk metrics, including concentration risk and risk distribution.
- **Stress Testing**: Simulate economic stress scenarios to evaluate portfolio resilience.
- **Regulatory Reporting**: Generate Basel III and IFRS 9 compliant reports.
- **AWS Integration**: Stream risk data to Kinesis and invoke SageMaker endpoints for ML predictions.

## Components

### 1. **Data Structures**
- **`CreditApplication`**: Represents a credit application with attributes like loan amount, credit score, and loan purpose.
- **`RiskMetrics`**: Contains calculated risk metrics such as PD, LGD, EAD, and expected loss.

### 2. **Risk Models**
- **`PDModel`**: Random Forest-based model for Probability of Default, with a fallback rule-based approach.
- **`LGDModel`**: Rule-based model for Loss Given Default, considering loan security and purpose.
- **`EADModel`**: Calculates Exposure at Default based on loan type and outstanding balance.

### 3. **Core Engine**
- **`CreditRiskEngine`**: Orchestrates risk assessment by integrating PD, LGD, and EAD models, and streams results to AWS Kinesis.

### 4. **Portfolio Analysis**
- **`PortfolioAnalyzer`**: Computes portfolio-level metrics, including total exposure, expected loss, and concentration risk.

### 5. **Stress Testing**
- **`StressTesting`**: Applies economic stress scenarios to assess portfolio vulnerability.

### 6. **Regulatory Reporting**
- **`RegulatoryReporting`**: Generates Basel III and IFRS 9 compliant reports based on portfolio analysis.

### 7. **AWS Services Integration**
- **`AWSServices`**: Handles data streaming to Kinesis and invokes SageMaker endpoints for ML predictions.
-
![architecture](./assets/credit-risk-app-arch.png)
-
## Installation

To set up the project, follow these steps:

```bash
git clone https://github.com/karthikvx/credit-risk-service-app.git
cd credit-risk-service-app
pip install -r requirements.txt
```

### To Run the Demo Without AWS:

No AWS Setup Needed: You don't need to configure AWS credentials or have an AWS account.
Run the Code: Simply execute python main.py as instructed.

### Important Notes:

Real-World Use: For a real-world deployment, you would need to:
Set up AWS credentials: Configure your AWS access key and secret key.
Create Kinesis Stream: Create a Kinesis stream named credit-risk-stream.
Deploy SageMaker Models: Train and deploy PD, LGD, and EAD models as SageMaker endpoints.
Mocked Data: Remember that the demo results are based on mock data and predictions. The actual performance of the system would depend on the quality of your trained models and real-world data.


## Usage

Run the demo script to see the system in action:

```bash
python main.py
```

### Example Output

```
=== Credit Risk Service App Demo ===

1. Individual Risk Assessment:
Application ID: APP000000
Loan Amount: $123,456.78
PD: 0.025 (2.5%)
LGD: 0.350 (35.0%)
EAD: $123,456.78
Expected Loss: $10,872.34
Risk Rating: BB
Confidence: 0.90

...

2. Portfolio Analysis:
Total Applications: 100
Total Exposure: $25,000,000.00
Total Expected Loss: $1,250,000.00
Portfolio Loss Rate: 0.050 (5.0%)
Average PD: 0.025
Risk Distribution: {'BB': 40, 'BBB': 30, 'A': 20, ...}

...

3. Stress Testing:
Mild Recession:
  Baseline Loss: $1,250,000.00
  Stressed Loss: $1,500,000.00
  Loss Increase: $250,000.00 (20.0%)

...

4. Regulatory Reporting:
Basel III Report (2023-10-01):
  Risk Weighted Assets: $25,000,000.00
  Capital Adequacy Ratio: 12.0%
  Expected Credit Losses: $1,250,000.00

...

=== Demo Complete ===
```

## Dependencies

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `boto3`
- AWS SDK for Python (Boto3)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [karthikvx@gmail.com](mailto:karthikvx@gmail.com).

---

**Note**: This README assumes the existence of a `requirements.txt` file and a `LICENSE` file in the repository. Adjust paths and details as necessary.