# Machine Learning and the Mind
This study strives to create a machine learning model to predict Alzheimer's disease in patients using their health data.

## Python Library Requirements
- pandas
- pyspark
- matplotlib
- sklearn
- tensorflow

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
WiP

## Recursive Feature Elimination Analysis
rfe_analysis.ipynb demonstrates feature analysis using recursive feature elimination (RFE). A simple neural network was trained with 32 different data sets, each having 1 of the 32 features of interest removed. The metrics from each model were compared to determine which features are the most useful. <br>

Here is the sequential model structure:
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |      31 | N/A                 |
| Hidden |       2 | Tanh                |
| Output |       1 | Sigmoid             |

## Supervised Machine Learning
The results of the recursive feature analysis guided the development of 3 models:

### Model 1 - All Features
This model used all 32 features from the dataset and served as a baseline. <br>

#### Structure
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |      32 | N/A                 |
| Hidden |       2 | Tanh                |
| Output |       1 | Sigmoid             |

#### Results
| Metric            | Value  | Epochs |
| ----------------- | ------ | ------ |
| Minimum Loss      | 38.24% |     40 |
| Maximum Accuracy  | 84.20% |     30 |
| Maximum Precision | 80.81% |     25 |
| Maximum Recall    | 74.48% |     40 |


### Model 2 - RFE Columns
This model used the 3 most siginificant features from RFE analysis: FunctionalAssessment, MemoryComplaints, and ADL. During the RFE analysis, the models these features were removed from performed the *worst* out of the 32 models. 

#### Structure
| Layer  | Neurons | Activation Function |
| ------ | ------- | ------------------- |
| Input  |       3 | N/A                 |
| Hidden |       5 | Tanh                |
| Output |       1 | Sigmoid             |

#### Results
Despite removing over 90% of the training data, this model performed almost as well as the baseline and even exceeded the baseline recall.
| Metric            | Value  | Epochs |
| ----------------- | ------ | ------ |
| Minimum Loss      | 42.20% |    175 |
| Maximum Accuracy  | 83.64% |    180 |
| Maximum Precision | 77.27% |     65 |
| Maximum Recall    | 76.72% |    180 |


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
| Minimum Loss      | 60.35% |     40 |
| Maximum Accuracy  | 70.26% |     60 |
| Maximum Precision | 68.82% |     60 |
| Maximum Recall    | 32.82% |     60 |
