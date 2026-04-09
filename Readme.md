# 🏦 EBO Early Buyout Optimization Engine

**Live App:**
👉 https://ebo-early-buyout-optimization-engine.streamlit.app/

---

## 📥 Sample Files

* [Download Sample Borrowers Data](SampleBrwData.csv)
* [Download Sample PMMS Data](historicalweeklydata.xlsx)

---

## 📌 Overview

The **EBO Early Buyout Optimization Engine** is a decision-support tool designed to identify delinquent FHA-insured loans that present the highest likelihood of successful re-performance following an Early Buyout (EBO), while ensuring favorable capital markets execution upon redelivery.

This tool combines **borrower behavior analytics** with **forward-looking market conditions** to enable smarter, risk-adjusted capital deployment.

---

## 🚀 Key Features

### 🔹 Dual Decision Framework

* Re-performance probability modeling
* Forward execution economics (redelivery gain/loss)

### 🔹 Configurable Policy Engine

* Delinquency window controls
* Risk thresholds
* Execution thresholds

### 🔹 Market-Aware Decisioning

* PMMS-based rate forecasting
* 120-day forward rate estimation

### 🔹 Explainability

* Loan-level factor breakdown
* Transparent scoring logic

---

## 📊 Data Inputs

### 🏠 Borrower & Loan Data

* LTV / CLTV
* Payment history & timing
* Delinquency status
* Income type & employment

### 📈 Market Data

* Freddie Mac PMMS historical rates
* Current + forward rate estimation

### 🌎 Macro Indicators

* Gas price trends
* Unemployment changes
* Home price index (HPI)

---

## 📤 Outputs

* Re-performance probability
* Partial Claim success probability
* Expected EBO success score
* Estimated redelivery gain/loss

### 📌 Recommendation Engine

* ✅ Execute EBO
* 👀 Monitor / Outreach
* ❌ Do Not EBO

---

## ⚙️ How to Run

```bash
git clone https://github.com/jayshane-wq/ebo-early-buyout-optimization-engine.git
cd ebo-early-buyout-optimization-engine
pip install -r requirements.txt
streamlit run ebo_early_buyout_prototype.py
```

---

## 🧪 How to Use

1. Upload:

   * `sample_borrowers.csv`
   * `sample_pmms.xlsx`

2. Turn OFF:

   * "Use sample candidate data"
   * "Use sample PMMS trend data"

3. Adjust:

   * Lookback window
   * Risk thresholds
   * Execution thresholds

4. Review:

   * Recommended candidates
   * Execution economics
   * Loan-level insights

---

## 💡 Strategic Value

* Improves EBO selection precision
* Aligns servicing decisions with capital markets
* Reduces re-default risk
* Enables dynamic strategy adjustment

---

## ⚖️ Regulatory Considerations

This tool is designed for **internal capital markets decisioning only**.

All borrowers:

* Receive consistent loss mitigation evaluation
* Follow identical servicing timelines
* Are not impacted by EBO selection logic

---

## 🔮 Future Enhancements

* Machine learning using Partial Claim performance data
* Real-time market pricing integration
* Scenario simulation (rate shocks, macro changes)
* Portfolio optimization dashboard

---

## ⚠️ Disclaimer

This is a prototype for demonstration purposes.
Execution pricing and borrower outcomes should be validated against production systems and regulatory guidelines.

---

## 👤 Author

**Jason Shane**
Operations & Product Transformation Leader
Mortgage Servicing | AI | Automation
