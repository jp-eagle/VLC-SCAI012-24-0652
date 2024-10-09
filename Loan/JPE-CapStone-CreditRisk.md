# CapStone Project Proposal for NTUC Module VLC-SCAI012-24-0652

## Project Description

### Objectives

To implement a Credit Risk Assessment Model for personal loans, it can be utilized to evaluate loan applications, determining both the approval status and the appropriate loan amount to be sanctioned.

### Project Type

Machine Learning Model for Credit Risk Assessment

### Problem to address

- The existing credit risk system is advanced and well-developed, but the cost of issuing loans remains high.
- Many individuals may seek loans but fail to meet the bank’s qualification criteria, even if they pose a low risk of default.
- Some borrowers might receive loans that surpass their actual requirements, potentially leading to unforeseen risks associated with the funds lent to them.
- Conducting timely re-evaluations of repayment capability is both expensive and time-consuming.

### The motivation for why you find this project interesting

- AI combined with big data can enable faster, lower-risk, and more cost-effective credit risk assessments.
- More people will benefit from easier, faster, and cheaper access to bank loans.

### Research any previous work that you know about

## Project Plan

1. Data understanding and preparation
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Development
5. Model Evaluation
6. Deployment
7. Conclusions and insights
   1. Summarize key finding
   2. Provide suggestions for future work.

## Datasets and its sample codes

1. [Loan.csv|20,000](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data), [Code1](https://www.kaggle.com/code/lorenzozoppelletto/financial-regression-and-binary-classification), [Code2](https://www.kaggle.com/code/jayrdixit/financial-risk-loan-approval)
2. [finance_dataset.csv|10,000](https://www.kaggle.com/datasets/kushagrakashyap23/finance-dataset)
3. [P39-Financial-Data.csv|17,908](https://www.kaggle.com/datasets/shubhi13/financial-dataset), [Source2](https://www.kaggle.com/datasets/dondata/loans-data)
4. [financial_risk_assessment.csv|15,000](https://www.kaggle.com/datasets/preethamgouda/financial-risk), [Code1](https://www.kaggle.com/code/preethamgouda/sample), [Code2](https://www.kaggle.com/code/vinod123kumar/finacial-risk), [Code3](https://www.kaggle.com/code/gouravgulia/financial-risk-assesment), [Code4](https://www.kaggle.com/code/kimkijun7/financial-risk-classifier-ml-ann-with-python), [Code5](https://www.kaggle.com/code/zeyadsayedadbullah/individual-financial-risk-analysis), [Code6](https://www.kaggle.com/code/mahmoudredagamail/financial-risk)
5. [credit_score_dataset.csv|1,000,000](https://www.kaggle.com/datasets/gautam02s/financial-record)

## References

1. [Learning Source1](https://www.youtube.com/watch?v=C3l92t0WmyQ&list=PLHPuG1bQvaJGTnmTp8nbNfEzcU9dT8jKQ&index=1)

## Dataset Study

| Field | Description |
| ----------- | ----------- |
|ApplicationDate|Loan Application Date|
|Age|Age, 18-80, in Years|
|AnnualIncome|Annual Income, in $|
|CreditScore|Credit Score, Credit Bureau issue, in number, [Introduction](https://www.creditbureau.com.sg/credit-score.html), [Singapore Personal Credit Report](https://www.creditbureau.com.sg/pdf/UYCR_Updated_25_July_2024.pdf), [Additional Sample](https://www.creditbureau.com.sg/pdf/Enhanced-Consumer-Credit-Report-2022.pdf)|
|EmploymentStatus|Employment Status, Employed/Self-Employed/unemployed|
|EducationLevel|Education Level, High School/ Associate/ Bachelor/ Master/ Doctorate|
|Experience| Number of Experiences, 0-61, in Years|
|LoanAmount|Amount to apply|
|LoanDuration|Duration to apply, 12-120, in Months|
|MaritalStatus| Marital Status, Single/ Married/ Divorced/ Widowed|
|NumberOfDependents|Number of Dependents, 0-5|
|HomeOwnershipStatus|Home Ownership, Own/ Mortgage/ Rent/ Other|
|MonthlyDebtPayments|Monthly Debt Payments|
|CreditCardUtilizationRate|Credit Card Utilization Rate|
|NumberOfOpenCreditLines|Number of Open Credit Line, 0-13, [Source](https://www.investopedia.com/terms/l/lineofcredit.asp)|
|NumberOfCreditInquiries|Number of Inquiries to Credit, 0-7, [Source](https://www.experian.com/blogs/ask-experian/how-many-hard-inquiries-is-too-many/)|
|DebtToIncomeRatio|??, Suspected it is wrongly Created, assume it is TotalDebtToIncomeRatio|
|BankruptcyHistory|Bankrupted before(1) or not(0)|
|LoanPurpose|Loan Purpose, Home/ Auto/ Education/ Debt Consolidation/ Other|
|PreviousLoanDefaults|Default before(1) or not(0)|
|PaymentHistory|8-45, in Month?|
|LengthOfCreditHistory|1-29, in Year?|
|SavingsAccountBalance|Saving Amount|
|CheckingAccountBalance|Cheque Amount|
|TotalAssets|Total Assets Value|
|TotalLiabilities|Total Liabilities Value|
|MonthlyIncome|= Annual Income / 12|
|UtilityBillsPaymentHistory|??|
|JobTenure|how long to work in current job, 0-16, in years|
|NetWorth|= Total Assets - Total Liabilities|
|BaseInterestRate|Starting Interest Rate|
|InterestRate|Applied Interest Rate|
|MonthlyLoanPayment|= Mortgage Calculation based on InterestRate, LoanAmount and LoanDuration, [Calculator](https://www.calculator.net/mortgage-calculator.html)|
|TotalDebtToIncomeRatio|= (MonthlyDebtPayments + MonthlyLoanPayment) / MonthlyIncome|
|LoanApproved|Loan Approved(1) or Rejected(0)|
|RiskScore|Risk Score, [Credit Score to Credit Risk Assessment](https://www.fibe.in/blogs/credit-score-vs-credit-risk-assessment-whats-the-difference/)|

### Preprocessing

1. Cleaning, missing, outlier, noise reduction
2. Transformation, standardize, feature Engineering(new/change current feature), categorize encoding(one-hot)
3. Balancing, to ensure different pattern of data has balance/enough records
4. Feature Selection, to rank the importance of features. and pick features that impact training most.
5. sampling, splitting, over/under sample data. to max model performance.

### Tunning Journey

- All Original fields
- One-Hot Mapping for all Object Feature Fields
- Convert Binary Field to Boolean Field
- Convert Number Field to Bin, and convert Bin into One-Hot, no impact
- KMeans to Gaussian Mixture
- Different Random State
- To analysis error records
- Remove Calculate Field, increase accuracy
- use Calculate Gini coefficient and true less than false, the accuracy increased.
- if use nest, i.e. use the false record to train again, the accuracy improved more. at end failed to use nest. result not good
- to use back the simple gini calc. and added no. of cluster and random state. then use 3d plot visualization to find out the accuracy. noticed that to use smaller simple gini (instead of biggest), can get a better result. but new problem is the Approved Rate is too low.

| Tuning | Accuracy |
| ----------- | ----------- |
|Original Fields and Values|73.9%|
|Nothing found in Correlation||


### Challenges

- after exclude outlier in training, the cluster is more extreme and validation is worse.
- to align the index of label after training is done
- to logic to determine which cluster belong to Approved or Rejected
- data problem, DebtToIncomeRatio and TotalDebtToIncomeRatio not consistently used. change: use TotalDebtToIncomeRatio only.
- Some obvious records should be rejected, but AI approved. to find reason

### Future Work

- Loan Amount Analysis
- Sophistic Validation Algorithm
- Further Error Analysis
- 