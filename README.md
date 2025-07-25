# Berkeley Professional Certificate in ML and AI : 2025 Final Capstone Project Part 2 : Module-24
### Project On: Detection And Prevention of Fradulent Phone Numbers
 
**Module 24** This Capstone Module closes this Final Capstone Project with its relevant complete solution and sumissions, executive summary included below.

#### Capstone Reference from the prior modules
**Module 6**  Drafting the Capstone Project Statment

**Module 16** About Finalizing the problem statement for the research project. 

**Module 20** Delved into the Dataset and its understanding and building baseline model for Capstone Project. This dataset with this module will be primarily used to analyze and build surrounding ML/AI Models which would help us in our predictions.

### Executive summary

This project focuses on identifying and preventing fraudulent phone numbers used during user registration or profile updates in enterprise Identity Access Systems (IAS), particularly targeting abuse of SMS-based authentication and verification mechanisms. Fraudsters exploit these systems using tactics such as VOIP numbers, synthetic identities, excessive OTP requests, and geolocation spoofing for monetary gain or unauthorized access. We are using a De-Identified dataset of 50,000 records with user metadata, phone behavior, registration patterns, and verification attributes. Please note this is a curtailed metadata considering the processing speed of the machine being used.

Using supervised machine learning we trained several models including Logistic Regression, Random Forest, XGBoost, and LightGBM. XGBoost delivered the best performance in terms of ROC-AUC, precision, and recall. SHAP analysis helped uncover key predictors such as the number of accounts tied to a phone number, OTP success rate, use of VOIP numbers, and IP-region mismatches. These features were consistently correlated with fraudulent behavior.

In parallel, unsupervised models like Isolation Forest and One-Class SVM were applied to identify anomalies without requiring fraud labels. Isolation Forest effectively surfaced suspicious users whose behaviors aligned with known fraud signatures. A combined strategy was adopted where high-confidence frauds were defined as those flagged by both supervised and unsupervised models.

Additional visualization techniques revealed that fraud was heavily concentrated among VOIP providers like Skype and Google Voice, and in certain geographic regions such as China, Nigeria, and Russia. Fraudulent users showed behavioral traits such as registering during odd hours, using unverified emails, and requesting multiple SMS verifications in a short time span.

The findings suggest that real-time detection models incorporating these insights can proactively block or verify high-risk registrations. Business recommendations include implementing dynamic fraud scoring during signup, flagging risky phone/IP patterns, and tightening SMS limits for VOIP numbers. This model-driven, data-informed approach provides a scalable foundation to enhance IAS security and defend against evolving fraud schemes.

# The Dataset

The Dataset used for this project is from my own Organization. The data is Anonymized and de-identified though these words are often used interchangeably. External users who are customers and partners register onto our organizations prime site seeking in IT NetWorking Solutions and Products. They typicall go thorugh a process of registration and there after come back to update their personal profile data through the profile pages after account login. Notifications during this process are primarily sent via Phones for validations. Some of the phone numbers are fraudulent and they are tapping in for no real business with our organization. Rouge users or Organizations accross geo locations exploit the SMS-Based systems for monetary gain and fraud. 

The Dataset used here is one such containing the listings of many users and their phone numbers now de-identified and anonymized. This is for keeping the confidentiality of the data from being spotted on the Organizational compliance radar.

#### Datafile Name - 🟨 Fraud_PhoneAnSMS_Dataset.csv 🟨

The original Dataset had a million rows. For keeping the Data size resonable for processing and based on the machine speeds and processing times involved the data size is curtailed to 50K and has 20 Columns.

# How is the project work and deliverables being measured? - Measurement Criteria Table As Below for Module-20 Capstone
<img width="799" height="690" alt="image" src="https://github.com/user-attachments/assets/95c7936e-46f9-4cda-a077-79553ffc84ea" />


# Note about the Files, Folder and Dataset...

-	The 🟢Code Folder will contain the Jupyter File.
- The 🟢Data Folder would contain both the Input data file at the root sourced for processing and modeling. The Data folder will contain the subfolder named "FraudPhoneNumber" which will contain the post analysis data of
  fradulent phone numbers found.
- The 🟢GraphPlots folder would contain 2 SubFolders - The root folder comntains all the plots related with EDS (Exploratory Data Analysis) and two subfolders : ⚪Supervised and ⚪Unsupervised. These subfolders will contain all the graphs for the respective models adopted in the project.

The Github place holder for this Project(Module-20) is 🟨 https://github.com/gurugrit/Finals_Capstone_UCB_MLAI_2025_MOD24 🟨

Folder Structure Snapshot:

<img width="430" height="264" alt="image" src="https://github.com/user-attachments/assets/b3cf96bd-8e84-4472-bc40-1f082799a747" />

# Phases of this Project
## 1. Business Understanding and Objective
Organizations that rely on phone numbers for identity verification and SMS-based systems (e.g., one-time passwords, referral programs, financial alerts) are increasingly targeted by fraudsters using fake, virtual, or reused phone numbers. These malicious actors exploit identity systems to:
- ✅ Bypass user verification steps
- ✅ Abuse promotional or referral schemes
- ✅ Automate mass registrations using VOIP or synthetic identities
- ✅ Launder accounts for spam, phishing, or monetization
This undermines trust, increases operational cost (e.g., via excessive SMS volume), and introduces risk to both users and the business.

**Business Objective**: The goal of this project is to detect and prevent fraudulent phone numbers at the point of user registration or profile update within the identity access system. By doing so, we aim to:
- ➡️ Reduce abuse of SMS-based systems (OTP, promo rewards)
- ➡️ Lower operational costs related to fraudulent SMS traffic
- ➡️ Improve onboarding quality and user trust
- ➡️ Prevent future fraudulent activities linked to bad actors
- ➡️ Enable scalable, data-driven fraud risk scoring

## 2. Data Understanding
### Data Description: The dataset contains 50000 entires with 20 Column-attributes/Features such as:

<img width="659" height="424" alt="image" src="https://github.com/user-attachments/assets/b8d97083-3a64-441b-8a1b-cab42bdc7771" />

## 3. Data Preparation
- We dropped 'user_id', 'email', 'phone_number', 'browser_fingerprint', 'ip_address'
- The Data File when observed did not contain any blanks or other irrelevant data which needed substitutions or transformations.
## 4. Modeling
🟨 **Planned and Proposed Supervised Models for this Project**
- 🕳️ Logistic Regression (Baselined around this)
- 🕳️ Random Forest
- 🕳️ XGBoost
- 🕳️ LightGBM
- 🕳️ KNN
- 🕳️ Naive Bayes:GaussianNB

🟨 **Planned and Proposed Unsupervised Models for this Project**
- 🕳️ Isolation Forest
- 🕳️ One-Class SVM
- 🕳️ LOF
- 🕳️ KMeans
  
## 5. Exploratory Data Analysis (Data Visualizations)

➡️ **Data Corelation Heatmap**
<img width="1019" height="911" alt="image" src="https://github.com/user-attachments/assets/8fafaaad-087d-41a4-9876-bb7b3e31f462" />

➡️ **Distribution of Key Numeric Features**
<img width="1381" height="886" alt="image" src="https://github.com/user-attachments/assets/95300f18-deac-4396-8893-f7287434ab3e" />

➡️ **Violin Plots of Numeric Features by Fraud Label**
<img width="1295" height="918" alt="image" src="https://github.com/user-attachments/assets/72880f2b-b406-4469-b6e3-8c463a0b3a49" />

➡️ **Kernel Density Estimate(KDE) Distributions of Features by Fraud Status**
<img width="1301" height="923" alt="image" src="https://github.com/user-attachments/assets/79e3c1c9-d0b8-4198-9dd3-b1ca9541ff64" />

➡️ **Violin Plots by Geo Region**
<img width="1298" height="925" alt="image" src="https://github.com/user-attachments/assets/c2147ff1-5ce0-46e2-90a6-686919f2de8d" />

## 6. Baseline Model Validation Summary

### Logistic Regression Confusion Matrix

<img width="1171" height="897" alt="image" src="https://github.com/user-attachments/assets/8e3c391f-d623-4802-a883-ac8c2bf46e02" />

- True Negatives : 9743
- False Positives: 71
- False Negatives: 100
- True Positives : 5086

<img width="413" height="284" alt="image" src="https://github.com/user-attachments/assets/03113595-ea10-4d8c-9442-a4a6e82b6320" />

From the above tables we can conclude that...
- Extremely high accuracy, precision, recall, and F1-scores.
- Balanced performance across both classes.
- Very few false positives/negatives.
- The dataset is also likely well-separated.

## ROC Curve Comparison:
The overall performance of a classifier, summarized over all possible classification thresholds, is given by the area under the ROC curve. An ideal ROC curve will hug the top left corner, 
indicating a high true positive rate and a low false positive rate; the larger the AUC( Area Under) the better the classifier.

While writing this readme. I had evaluated all the models I had planned to use, the ROC Curve for which is depicted below. This is also to baseline this along with the Baseline model it was being evaluated against.

<img width="1051" height="627" alt="image" src="https://github.com/user-attachments/assets/11486151-4cbd-4b49-b967-7e963fe08d08" />

## 🟢 Further work Related with this Project Continues with Module 24 




