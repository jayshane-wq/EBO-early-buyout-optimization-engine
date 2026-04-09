EBO Early Buyout Optimization Engine
  
  App: https://ebo-early-buyout-optimization-engine.streamlit.app/

## 📥 Sample Files

- [Download Sample Borrowers Data File](SampleBrwData.csv)
- [Download Sample PMMS Data](historicalweeklydata.xlsx)

Overview

The EBO Early Buyout Optimization Engine is a decision-support tool designed to identify delinquent FHA-insured loans that present the highest likelihood of successful re-performance following an Early Buyout (EBO), while ensuring favorable capital markets execution upon redelivery.

This tool combines borrower behavior analytics with forward-looking market conditions to enable smarter, risk-adjusted capital deployment.

Key Features
🔹 Dual Decision Framework
Re-performance probability modeling
Forward execution economics (redelivery gain/loss)
🔹 Configurable Policy Engine
Delinquency window controls
Risk thresholds
Execution thresholds
🔹 Market-Aware Decisioning
PMMS-based rate forecasting
Current vs forward pricing simulation
🔹 Explainability
Loan-level factor breakdown
Transparent scoring logic
Inputs
Loan & Borrower Data
LTV / CLTV
Payment history
Delinquency status
Income type & employment
Market Data
Freddie Mac PMMS historical rates
120-day forward rate estimation
Macro Indicators
Gas price trends
Unemployment changes
Home price index (HPI)
Outputs
Re-performance probability
Partial Claim success probability
Expected EBO success score
Estimated redelivery gain/loss
Recommendation:
Execute EBO
Monitor / Outreach
Do Not EBO
Installation
1. Clone repository
git clone https://github.com/YOUR_USERNAME/ebo-optimization-engine.git
cd ebo-optimization-engine
2. Install dependencies
pip install -r requirements.txt
3. Run application
streamlit run ebo_early_buyout_prototype.py
Usage
Upload candidate loan dataset (CSV)
Upload Freddie Mac PMMS file (Excel or CSV)
Adjust policy levers:
Lookback window
Risk thresholds
Execution thresholds
Review:
Recommended EBO candidates
Execution economics
Borrower-level explainability
Strategic Value
Improves EBO selection precision
Aligns servicing decisions with capital markets
Reduces re-default exposure
Enables dynamic strategy adjustment
Regulatory Considerations

This tool is designed for internal capital markets decisioning only.

All borrowers:

Receive consistent loss mitigation evaluation
Follow identical servicing timelines
Are not impacted by EBO selection logic
Future Enhancements
Machine learning model using Partial Claim performance data
Real-time market pricing integration
Scenario simulation (rate shocks, macro changes)
Portfolio optimization dashboard
Disclaimer

This is a prototype for demonstration and strategy development purposes.
Execution pricing and borrower outcomes should be validated against production systems and regulatory guidelines.

Author

Jason Shane
Operations & Product Transformation Leader
Mortgage Servicing | AI | Automation
