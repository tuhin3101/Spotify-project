<p align="center">
  <img src="https://user-images.githubusercontent.com/121638166/228399769-4fa46a44-2cc9-4113-8ac5-d260b9d1fe73.png">
</p>


# Project-Bondora-Financial-risk-modelling-P2P

The main purposes of this analysis are to summarize the characteristics of variables that can affect the loan status and to get some ideas about the relationships among variables.


## Project Summary

 # Project-Bondora-Financial-risk-modelling-of-European-P2P-lending-platform
The main purposes of this analysis are to summarize the characteristics of variables that can affect the loan status and to get some ideas about the relationships among variables.


This project is a collaboration between our team that worked on developing a machine learning model using Gradient Boosting and VSM (Vector Space Model). We aimed to achieve the highest accuracy in our model by comparing our data and performing different EDA (Exploratory Data Analysis) techniques on our target variable.




### Abstract
In this project, we will model the credit risk of Bondora systems for peer-to-peer lending. A major European P2P lending platform provided the study's data (Bondora). A pool of defaulted and non-defaulted loans from the years 1 March 2009 and 27 January 2020 make up the data that was retrieved. The database includes loan transactions as well as demographic and financial details about the borrowers. Loans in P2P lending are frequently made without any kind of collateral, and lenders want for larger returns to make up for the financial risk they are taking. Additionally, they must make choices that benefit the borrowers despite knowledge asymmetry. Lenders strive to reduce the risk of default associated with each loan decision and realise the return that makes up for the risk in order to make logical decisions.
### Background of Understanding the Problem
Peer-to-peer lending has received a lot of attention recently, partly because it provides a cutting-edge method of bringing together borrowers and lenders. Yet there's more to it, just like with other cutting-edge commercial strategies. Many might ask, for instance, what makes peer-to-peer lending so different from dealing with a bank—or, perhaps, why it is so much better—or why it has gained popularity in so many regions of the world.
Indeed, the sector has experienced significant expansion in recent years. According to Business Insider, the world's two largest P2P markets, the U.S. and Europe, have seen transaction volumes grow at double and, in some cases, triple-digit percentage rates thanks to a legal climate that is friendly to online commerce and popular acceptance of it.
Peer-to-peer lending, or "P2P," presents an alluring opportunity for investors to diversify portfolios and improve long-term performance. Via peer-to-peer platforms, they can profit from an asset class that has proven successful in both prosperous and difficult times. They can also avoid the hazards of placing all of their eggs in one basket, which is crucial at a time when many experts think that traditional investments like stocks and bonds are more dangerous than ever.
In peer-to-peer (P2P) lending, default risk has long been an important risk factor to evaluate borrower behaviour. Loans made through peer-to-peer lending are frequently uncollateralized, and lenders want for higher returns to make up for the financial risk they are taking. Additionally, they must make choices that benefit the borrowers despite knowledge asymmetry. Lenders strive to minimise the risk of default associated with each loan decision and realise the return that accounts for the risk in order to make informed decisions.
Similar to the field of financial research, there aren't many datasets that can be used for developing and examining credit risk models. The research community will benefit from this dataset as it develops and conducts research in the area of credit risk.
#### Reasons why a loan could be rejected:
Many factors go into determining eligibility for a personal loan. The most common reasons for rejections are as follows,

*	Missing important information/paperwork
*	Low credit score or bad credit history 
*	A high debt-to-income ratio
*	Unstable employment history 
*	Too low of income for the desired loan amount 
*	Loan purpose didn’t meet the lender’s criteria 

## Data Wrangling
### Data Details

*	Data has 134529 records and 112 columns.
*	Data has no duplicates.
*	Data has missing values.

### Data Cleaning

•	First, we'll remove the unnecessary columns.

```python
drop_cols=['BidsApi', 'BidsPortfolioManager', 'BidsPortfolioManager', 'BidsManual',
           'CurrentDebtDaysPrimary', 'CurrentDebtDaysPrimary', 'PrincipalOverdueBySchedule', 
           'PrincipalOverdueBySchedule', 'PrincipalOverdueBySchedule', 'IncomeFromPrincipalEmployer',
           'IncomeFromPension', 'EAD1', 'EAD2', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
           'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther', 'ReportAsOfEOD', 'LoanId','LoanNumber',
           'ListedOnUTC','DateOfBirth','BiddingStartedOn','UserName','LoanApplicationStartedDate','FirstPaymentDate',
           'LoanApplicationStartedDate','ApplicationSignedHour', 'ApplicationSignedWeekday',
           'ActiveScheduleFirstPaymentReached','ModelVersion','WorseLateCategory','PlannedPrincipalTillDate',
         'ProbabilityOfDefault', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn', 'GracePeriodStart', 'GracePeriodEnd',
           'NextPaymentDate', 'NextPaymentNr', 'PreviousEarlyRepaymentsBefoleLoan', 'PreviousRepaymentsBeforeLoan',
           "NrOfScheduledPayments","ReScheduledOn","PrincipalDebtServicingCost","InterestAndPenaltyDebtServicingCost",
           "ActiveLateLastPaymentCategory", 'PlannedInterestTillDate', 'CurrentDebtDaysSecondary',
           'PlannedPrincipalPostDefault', 'PlannedInterestPostDefault', 'PrincipalRecovery',
           'InterestRecovery', 'RecoveryStage', 'EL_V0', 'EL_V0', 'NrOfDependants', 'Rating_V0', 'CreditScoreEsEquifaxRisk', 
          'CreditScoreFiAsiakasTietoRiskGrade', 'GracePeriodStart','GracePeriodEnd', 'CreditScoreEeMini', 'PrincipalPaymentsMade',
           'MonthlyPaymentDay','LastPaymentOn', 'DebtOccuredOn', 'DebtOccuredOnForSecondary', 'DefaultDate', 'StageActiveSince', 'EL_V1', 
           'Rating_V1', 'Rating_V2', 'ActiveLateCategory','CreditScoreEsMicroL', 'NewCreditCustomer','LoanDate',
           'NewCreditCustomer','LoanDate','ContractEndDate','MaturityDate_Original','MaturityDate_Last', 'ContractEndDate', 'MaturityDate_Original', 'MaturityDate_Last'
                                  , 'Restructured', 'InterestAndPenaltyPaymentsMade', 
                                  'PrincipalWriteOffs', 'InterestAndPenaltyWriteOffs', 
                                  'PrincipalBalance', 'InterestAndPenaltyBalance',  
                                  'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan', 
                                  'PreviousEarlyRepaymentsCountBeforeLoan', 'VerificationType', 'LanguageCode',
                                   'LoanDuration', 'OccupationArea', 'ExistingLiabilities', 'LiabilitiesTotal',
                                   'RefinanceLiabilities', 'EmploymentDurationCurrentEmployer', 'County', 'City', 'Country', 'AppliedAmount', 'Rating', 'EmploymentPosition', 'FreeCash']
print(len(drop_cols))
modifed_data.drop(drop_cols,axis=1,inplace=True)
```

•	Now, we'll remove the null values.



```python
print(modifed_data.isnull().sum())

# drop rows with null values
modifed_data.dropna(inplace=True)
```
### Exploratory Data Analysis (EDA)

In EDA, we have examined a number of columns to learn more about the provided dataset.

#### Target (Status):

We have 3 unique values in the Status columns Repaid, Late and Current. 
* Repaid: Loan repayment is the act of settling an amount borrowed from a lender along with the applicable interest amount.
* Late: The loan has one or more interest payments which are late. Overdue – there has been a delay of more than a day on the scheduled repayment date of the loan.
* Current: Current means The borrower is making payments on time. Grace period is a set number of days after the due date during which payment may be made by the borrower without penalty.
We have removed the status "Current".

#### Gender:
We looked at the gender distribution, and the bar chart plainly demonstrates that roughly 60% of users are male and 40% are female.


##### To obtain a better understanding of the Data, we also looked at the distribution of Education, Marital Status, Employment duration, Employment status, and other factors.

## EDA graphs:

![7g886j](https://user-images.githubusercontent.com/121638166/228406704-457fb7e3-1b14-4fa6-b983-123d79d710b9.gif)

## Comparison graphs with Status:

![7g88ls](https://user-images.githubusercontent.com/121638166/228407233-846b3f7f-b131-4cbe-8a1a-118276d9ef66.gif)




## Modeling

### Gradient Boosting classifier

<img width="410" alt="image" src="https://user-images.githubusercontent.com/123512564/221941696-afda8083-3ef1-4430-85dd-c3b2b036e533.png">


### Super vector machine model


<img width="407" alt="image" src="https://user-images.githubusercontent.com/123512564/221941466-3125d410-f054-4ee5-ae1c-fbcba4d1705e.png">



### Creating the pipeline

We have created a .pkl pipelines to create a classification and reggresion pipelines


### Deployng the model usinf PythonAnywhere

We have created a Flask app to connect it with the front-end page:

http://23bondora23.pythonanywhere.com/

![FsYbHxzXoAEnBpk](https://user-images.githubusercontent.com/123512564/228510418-0012af27-4d94-4435-aef3-c3f320955251.jpg)

### The team work

Here is a breakdown of the roles and responsibilities of each team member:


| Team member name         | Role and Tasks|
| :-------------: | :-------------: |
| Fatimah | - Data preprocessing <br> - Feature engineering <br> - Modeling (Classification, Regression) <br> - Pipeline creation <br> - Model Deployment <br>|
| Chebrolu | Data cleaning and preprocessing, and feature engineering  |
| Ashish | Data preprocessing, feature engineering and modeling |
| Asif | Data analysis and preprocessing |
| Mohamed | Data analysis and preprocessing |






