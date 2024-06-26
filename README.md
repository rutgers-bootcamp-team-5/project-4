# Machine Learning and the Mind
This study strives to create a machine learning model to predict Alzheimer's disease in patients using their health data.

## Python Library Requirements
- pandas
- pyspark
- matplotlib
- sklearn
- tensorflow

## Usage
**data** - This directory contains the data used for analysis and modeling. <br>
**dev** - This directory contains code snippets used during development. <br>
**main.ipynb** - This notebook presents the main data analysis and modeling. <br>
**rfe_analysis.ipynb** - This notebook demonstrates a recursive feature elimination strategy that guided model development. Execution time can last in excess of 20 minutes.

## Dataset
The dataset contains extensive health data for 2149 patients. The data is divided into categories and listed below.

### Source
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data

### Column Descriptions 

#### Patient Information
**PatientID**: A unique identifier assigned to each patient (4751 to 6900).

#### Demographic Details
**Age**: The age of the patients ranges from 60 to 90 years.<br>
**Gender**: Gender of the patients, where 0 represents Male and 1 represents Female.<br>
**Ethnicity**: The ethnicity of the patients, coded as follows:<br>
&nbsp;&nbsp;&nbsp;&nbsp;  0: Caucasian <br>
&nbsp;&nbsp;&nbsp;&nbsp;  1: African American <br>
&nbsp;&nbsp;&nbsp;&nbsp;  2: Asian <br>
&nbsp;&nbsp;&nbsp;&nbsp;  3: Other <br>
**EducationLevel**: The education level of the patients, coded as follows: <br>
&nbsp;&nbsp;&nbsp;&nbsp;  0: None <br>
&nbsp;&nbsp;&nbsp;&nbsp;  1: High School <br>
&nbsp;&nbsp;&nbsp;&nbsp;  2: Bachelor's <br>
&nbsp;&nbsp;&nbsp;&nbsp;  3: Higher 
  
#### Lifestyle Factors
**BMI**: Body Mass Index of the patients, ranging from 15 to 40.<br>
**Smoking**: Smoking status, where 0 indicates No and 1 indicates Yes.<br>
**AlcoholConsumption**: Weekly alcohol consumption in units, ranging from 0 to 20.<br>
**PhysicalActivity**: Weekly physical activity in hours, ranging from 0 to 10.<br>
**DietQuality**: Diet quality score, ranging from 0 to 10.<br>
**SleepQuality**: Sleep quality score, ranging from 4 to 10.

#### Medical History
**FamilyHistoryAlzheimers**: Family history of Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.<br>
**CardiovascularDisease**: Presence of cardiovascular disease, where 0 indicates No and 1 indicates Yes.<br>
**Diabetes**: Presence of diabetes, where 0 indicates No and 1 indicates Yes.<br>
**Depression**: Presence of depression, where 0 indicates No and 1 indicates Yes.<br>
**HeadInjury**: History of head injury, where 0 indicates No and 1 indicates Yes.<br>
**Hypertension**: Presence of hypertension, where 0 indicates No and 1 indicates Yes.

#### Clinical Measurements
**SystolicBP**: Systolic blood pressure, ranging from 90 to 180 mmHg.<br>
**DiastolicBP**: Diastolic blood pressure, ranging from 60 to 120 mmHg.<br>
**CholesterolTotal**: Total cholesterol levels, ranging from 150 to 300 mg/dL.<br>
**CholesterolLDL**: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.<br>
**CholesterolHDL**: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.<br>
**CholesterolTriglycerides**: Triglycerides levels, ranging from 50 to 400 mg/dL.

#### Cognitive and Functional Assessments
**MMSE**: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.<br>
**FunctionalAssessment**: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.<br>
**MemoryComplaints**: Presence of memory complaints, where 0 indicates No and 1 indicates Yes.<br>
**BehavioralProblems**: Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.<br>
**ADL**: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.

#### Symptoms
**Confusion**: Presence of confusion, where 0 indicates No and 1 indicates Yes.<br>
**Disorientation**: Presence of disorientation, where 0 indicates No and 1 indicates Yes.<br>
**PersonalityChanges**: Presence of personality changes, where 0 indicates No and 1 indicates Yes.<br>
**DifficultyCompletingTasks**: Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes.<br>
**Forgetfulness**: Presence of forgetfulness, where 0 indicates No and 1 indicates Yes.

#### Diagnosis Information
**Diagnosis**: Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.

#### Confidential Information
**DoctorInCharge**: This column contains confidential information about the doctor in charge, with "XXXConfid" as the value for all patients.

## Exploratory Data Analysis
Analysis of all the data provided. Created graphs about the particements of the study. Look at the
1) Number of Patient in the Study with Diagnosis.
2) Participent BMI
3) Participent Age
4) Participent Gender
5) Participent Ethnicity
6) Participent History of Head Injusy
7) Participent Family History of Alzheimers
8) Participent Dyagnois
9) Scatter Plot Diet Quality vs BMI

## Unsupervised Machine Learning
Model exploration began with clustering using k-means. There are more than 30 columns in dataset, so PCA was implemented to reduce number of dimension with loss of about 2% data. Please refer to the main notebook for the elbow curve and scatter plot for clusters.

## Supervised Machine learning - KNN
The first iteration of supervised learning used the KNN (K-nearest neighbour) algorithm. A confusion matrix was derived to determine false positive and false negatives. The model accuracy reached only 71%, so other machine learning methods were pursued.
    
## Recursive Feature Elimination Analysis
rfe_analysis.ipynb demonstrates feature analysis using recursive feature elimination (RFE). A simple neural network was trained with 32 different data sets, each having 1 of the 32 features of interest removed. The metrics from each model were compared to determine which features are the most useful. <br>

Here is the sequential model structure:
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |      31 | N/A                 |
| Hidden |       2 | Tanh                |
| Output |       1 | Sigmoid             |

## Supervised Machine Learning - Neural Networks
The results of the recursive feature analysis guided the development of 3 models:

### Model 1 - All Features
This model used all 32 features from the dataset and served as a baseline. 

#### Structure
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |      32 | N/A                 |
| Hidden |       2 | Tanh                |
| Output |       1 | Sigmoid             |

#### Results
| Metric            | Value  | Epochs |
| ----------------- | ------ | ------ |
| Minimum Loss      | 41.54% |     30 |
| Maximum Accuracy  | 82.16% |     75 |
| Maximum Precision | 81.70% |     10 |
| Maximum Recall    | 72.82% |     75 |


### Model 2 - RFE Columns
This model used the 3 most siginificant features from RFE analysis: FunctionalAssessment, MemoryComplaints, and ADL. During the RFE analysis, the models these features were removed from performed the *worst* out of the 32 models. 

#### Structure
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |       3 | N/A                 |
| Hidden |       5 | Tanh                |
| Output |       1 | Sigmoid             |

#### Results
Despite removing over 90% of the training data, this model surpassed the baseline in all metrics.
| Metric            | Value  | Epochs |
| ----------------- | ------ | ------ |
| Minimum Loss      | 40.94% |    155 |
| Maximum Accuracy  | 85.69% |    260 |
| Maximum Precision | 82.80% |     70 |
| Maximum Recall    | 79.40% |    260 |


### Model 3 - Non-RFE Columns
This model used the 29 features not included in Model 2.

#### Structure
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |      29 | N/A                 |
| Hidden |       2 | Tanh                |
| Output |       1 | Sigmoid             |

#### Results
There is a noticeable decrease in performance in this model, further supporting the importance of the RFE columns.
| Metric            | Value  | Epochs |
| ----------------- | ------ | ------ |
| Minimum Loss      | 63.07% |     25 |
| Maximum Accuracy  | 67.66% |     70 |
| Maximum Precision | 58.82% |     70 |
| Maximum Recall    | 35.90% |     70 |
