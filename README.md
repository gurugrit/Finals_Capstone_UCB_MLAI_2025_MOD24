# Berkeley Professional Certificate in ML and AI : 2025 Final Capstone Project Part 2 : Module-24
### Project On: Detection And Prevention of Fradulent Phone Numbers

### 🟢 Executive summary

This project focuses on identifying and preventing fraudulent phone numbers used during user registration or profile updates in enterprise Identity Access Systems (IAS), particularly targeting abuse of SMS-based authentication and verification mechanisms. Fraudsters exploit these systems using tactics such as VOIP numbers, synthetic identities, excessive OTP requests, and geolocation spoofing for monetary gain or unauthorized access. We are using a De-Identified dataset of 50,000 records with user metadata, phone behavior, registration patterns, and verification attributes. Please note this is a curtailed metadata considering the processing speed of the machine being used.

Using supervised machine learning we trained several models including Logistic Regression, Random Forest, XGBoost, and LightGBM. XGBoost delivered the best performance in terms of ROC-AUC, precision, and recall. SHAP analysis helped uncover key predictors such as the number of accounts tied to a phone number, OTP success rate, use of VOIP numbers, and IP-region mismatches. These features were consistently correlated with fraudulent behavior.

In parallel, unsupervised models like Isolation Forest and One-Class SVM were applied to identify anomalies without requiring fraud labels. Isolation Forest effectively surfaced suspicious users whose behaviors aligned with known fraud signatures. A combined strategy was adopted where high-confidence frauds were defined as those flagged by both supervised and unsupervised models.

Additional visualization techniques revealed that fraud was heavily concentrated among VOIP providers like Skype and Google Voice, and in certain geographic regions such as China, Nigeria, and Russia. Fraudulent users showed behavioral traits such as registering during odd hours, using unverified emails, and requesting multiple SMS verifications in a short time span, orgination of traffic from rouge IP's and possible repetitive attempts suspected to be thorugh deployed BOT agents.

The findings suggest that real-time detection models incorporating these insights can proactively block or verify high-risk registrations. Business recommendations include implementing dynamic fraud scoring during signup, flagging risky phone/IP patterns, and tightening SMS limits for VOIP numbers. This model-driven, data-informed approach provides a scalable foundation to enhance IAS security and defend against evolving fraud schemes.

#### Capstone Reference from the prior modules
🟡 **Module 6**  Drafting the Capstone Project Statment

🟡 **Module 16** About Finalizing the problem statement for the research project. 

🟡 **Module 20** Delved into the Dataset,  its understanding and building baseline model for Capstone Project. This dataset with this module will be primarily be used to analyze and build surrounding ML/AI Models which would help us in our predictions.        https://github.com/gurugrit/Capstone_UCB_MLAI_2025

# The Dataset

The Dataset used for this project is from my own Organization. The data is Anonymized and De-identified though these words are sometimes used interchangeably. External users who are customers and partners register onto our organizations prime site seeking in IT NetWorking Solutions and Products. They typicall go thorugh a process of registration and there after come back to update their personal profile data through the profile pages after account login. Notifications during this process are primarily sent via Phones for validations. Some of the phone numbers are fraudulent and the users behind have no real business with our organization. Rouge users or Organizations accross geo locations exploit the SMS-Based systems for monetary gain and fraud. 

The Dataset used here is one such containing the listings of many users and their phone numbers now de-identified and anonymized. This is for keeping the confidentiality of the data from being spotted on the Organizational compliance radar.

### Imabalanced and Balanced Dataset. Do We Need to have Balanced Data?
The dataset has 30-35% fraud – it’s not extremely imbalanced.
Models like XGBoost and Random Forest handle class imbalance well (using scale_pos_weight).
For model training (especially for SHAP, unsupervised learning, visualizations), having that kind of percetages of fraud gives:
🔺Enough examples to learn patterns 🔺Better visual and statistical separation 🔺Richer evaluation of precision-recall trade-offs

#### Datafile Name - 🟨 Fraud_PhoneAnSMS_Dataset.csv 🟨 
The organization of the Folder structure in Git is as depicted below in the Files Folder Section of this Readme.

The original Dataset had a million rows. For keeping the Data size resonable for processing and based on the machine speeds and processing times involved the data size is curtailed to 50K and has 20 Columns.

# How is the project work and deliverables being measured? - Measurement Criteria Table As Below for Module-20 Capstone
<img width="799" height="690" alt="image" src="https://github.com/user-attachments/assets/95c7936e-46f9-4cda-a077-79553ffc84ea" />


# Note about Project Folder, Files, and Dataset...

-	The 🟢Code Folder contains the Jupyter File. The executed code itself has comments captured and with related Term definitions at places and outputs in the uploaded Jupyter file. 
- The 🟢Data Folder contains both the Input data file at the root, sourced for processing and modeling. The Data folder also contains the subfolder named "FraudPhoneNumber" containing the post analysis data of
  fradulent phone numbers found after the Jupyter file is executed in its entirety.
- The 🟢GraphPlots folder contains 2 SubFolders - The root folder which comntains all the plots related with EDS (Exploratory Data Analysis) and two subfolders : ⚪Supervised and ⚪Unsupervised. These subfolders will contain all the graphs for the respective models adopted in the project.

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
- 
This undermines trust, increases operational cost (e.g., via excessive SMS volume), and introduces risk to both users and the business.

**Business Objective**: The goal of this project is to detect and prevent fraudulent phone numbers at the point of user registration or profile update within the identity access system. By doing so, we aim to:

- ➡️ Reduce abuse of SMS-based systems during (OTP, promo rewards/invites, Registrations/Profile)
- ➡️ Lower operational costs related to fraudulent SMS traffic
- ➡️ Improve onboarding quality and user trust
- ➡️ Prevent future fraudulent activities linked to bad actors
- ➡️ Enable scalable, data-driven fraud risk scoring

## 2. Data Understanding
### Data Description: The dataset contains 50000 entires with 20 Column-attributes/Features such as:

<img width="659" height="424" alt="image" src="https://github.com/user-attachments/assets/b8d97083-3a64-441b-8a1b-cab42bdc7771" />

## 3. Data Preparation
- We dropped 'user_id', 'email', 'phone_number', 'browser_fingerprint', 'ip_address'
- The Data File when observed did not contain any blanks or other irrelevant data which needed larger specific substitutions or transformations. However we did define...
  
🔹Pipeline SimpleImputer - StandardScalar
🔹Pipeline SimpleImputer - OneHotEncoder
🔹ColumnTransformer

The above three statements defined the data preprocessing pipeline using scikit-learn, separating the treatment of numeric and categorical features before feeding them into a machine learning model. This three-step setup builds a robust preprocessing system to handle missing values, scaling, and encoding, keeping ML pipeline clean, reproducible, and production-ready.
  
## 4. Modeling
This Capstone project is modelled around Supervised and Unsupervised modeling algorithms of ML.

🟨 **Supervised Models for this Project**
- 🕳️ Logistic Regression (Baselined around this)
- 🕳️ Random Forest
- 🕳️ XGBoost (eXtreme Gradient Boosting)
- 🕳️ LightGBM (Light Gradient-Boosting Machine)
- 🕳️ KNN (k-nearest neighbors algorithm)
- 🕳️ Naive Bayes:GaussianNB (Gaussian Naive Bayes (GNB) is a classification technique used in machine learning based on a probabilistic approach and Gaussian distribution

🟨 **Unsupervised Models for this Project**
- 🕳️ Isolation Forest
- 🕳️ One-Class SVM (One-Class Support Vector Machines)
- 🕳️ LOF (Local Outlier Factor) for Anomaly Detection
- 🕳️ KMeans (Vector Quantization Technique, has somewhat loose relationship to KNN) 
  
## 5. Exploratory Data Analysis (Data Visualizations)

➡️ **Data Corelation Heatmap**
<img width="1021" height="913" alt="image" src="https://github.com/user-attachments/assets/3d06c43c-bec7-4da3-956c-65d9918151e9" />

➡️ **Distribution of Key Numeric Features**
<img width="1374" height="891" alt="image" src="https://github.com/user-attachments/assets/645fc0c6-d98e-41f5-8976-71fa47875be7" />

➡️ **Violin Plots of Numeric Features by Fraud Label**
<img width="1301" height="921" alt="image" src="https://github.com/user-attachments/assets/5bc10e60-b662-4857-87fd-af8c32de1941" />

➡️ **Kernel Density Estimate(KDE) Distributions of Features by Fraud Status**
<img width="1303" height="923" alt="image" src="https://github.com/user-attachments/assets/ed1afacf-be55-422a-b4fd-94f0c5b9cee8" />

➡️ **Violin Plots by Geo Region**
<img width="1304" height="923" alt="image" src="https://github.com/user-attachments/assets/24c14b87-c41d-4c82-abe0-05dba2af1a18" />

➡️ **Violin Plots by Device (ios, web, android) Type**
<img width="1302" height="924" alt="image" src="https://github.com/user-attachments/assets/d88e8e89-3b95-4aaa-9a54-2f60e9140f65" />

➡️ **Bar Plots by Provider Distribution (Google, skype, Twilio) Type**
<img width="1840" height="898" alt="image" src="https://github.com/user-attachments/assets/1a4e87d1-8db7-44e9-a6b7-5799675435da" />

https://github.com/gurugrit/Finals_Capstone_UCB_MLAI_2025_MOD24/blob/main/GraphPlots/EDS-Correlation_Heatmap.png

## 6. Supervised Model Validation Summaries and Conclusion on the Best Model

### 🟡Logistic Regression Confusion Matrix

- True Negatives : 9743
- False Positives: 71
- False Negatives: 100
- True Positives : 5086

<img width="1175" height="901" alt="image" src="https://github.com/user-attachments/assets/1b262224-3d34-477a-acf5-8a5660f71b47" />

From the above tables we can conclude that...
- Extremely high accuracy, precision, recall, and F1-scores.
- Balanced performance across both classes.
- Very few false positives/negatives.
- The dataset is also likely well-separated.

 ### 🟡Random Forest Confusion Matrix

<img width="1179" height="902" alt="image" src="https://github.com/user-attachments/assets/dce977b0-d8d1-47f2-8799-24e764a8e440" />

### 🟡XGBoost Confusion Matrix

<img width="1181" height="907" alt="image" src="https://github.com/user-attachments/assets/f3ced31d-ad09-49fe-a423-fad54fb5fe8e" />

### 🟡LightGBM Confusion Matrix

<img width="1182" height="902" alt="image" src="https://github.com/user-attachments/assets/b4c54350-6545-4531-a657-e3d214f374d2" />

### 🟡KNN Confusion Matrix

<img width="1170" height="899" alt="image" src="https://github.com/user-attachments/assets/3b1adffb-8547-44db-9795-92ab3cc3c97d" />

### 🟡Naive Bayes Confusion Matrix

<img width="1183" height="906" alt="image" src="https://github.com/user-attachments/assets/e234e891-cce5-470b-82b5-bb0ff99dc2ef" />

## ROC Curve Model Comparisons:
The overall performance of a classifier, summarized over all possible classification thresholds, is given by the area under the ROC curve. An ideal ROC curve will hug the top left corner, 
indicating a high true positive rate and a low false positive rate; the larger the AUC( Area Under) the better the classifier.

<img width="1537" height="908" alt="image" src="https://github.com/user-attachments/assets/d0f45b9f-577b-488d-b648-db9b6f234ac4" />

## 🟨Feature Importance - Model Comparisons:
<img width="1911" height="753" alt="image" src="https://github.com/user-attachments/assets/bd8f2bb4-b325-4680-8982-64fd87b8ac58" />

## 🟨Model Performance Plot
<img width="1537" height="903" alt="image" src="https://github.com/user-attachments/assets/a5132b5c-9124-436d-8f85-01455a87a6c1" />

## ✅ Conclusion on the Best Model

<img width="362" height="146" alt="image" src="https://github.com/user-attachments/assets/8c3d02e5-dd3f-4f98-b57f-a168f4f5c110" />

## 7. Unsupervised Model Validation Summaries and Conclusion on the Best Model
## 🟪Models Before Tuning - ROC Curve
<img width="1526" height="904" alt="image" src="https://github.com/user-attachments/assets/e89ffd64-37e4-4abd-ac64-ae1d3098e0b9" />

<img width="479" height="187" alt="image" src="https://github.com/user-attachments/assets/6a19717d-c1d0-4aaf-9fe5-d8b91a88c25d" />

## 🟪Models After Tuning - ROC Curve
This ROC curve plot shows how well different unsupervised anomaly detection models perform after hyperparameter tuning. The curves compare models by plotting their True Positive Rate (TPR) vs. False Positive Rate (FPR), and the Area Under the Curve (AUC) metric summarizes their corresponding effectiveness.
<img width="1232" height="717" alt="image" src="https://github.com/user-attachments/assets/b5ec806f-849c-47aa-95b5-b2ef1a593ba2" />

## 🔍Before and After Tuning comparisons...
<img width="567" height="169" alt="image" src="https://github.com/user-attachments/assets/c6cbb597-1d27-42cd-b13e-499edc0a2c34" />




