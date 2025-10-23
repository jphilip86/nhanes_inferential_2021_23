# NHANES 2021-2023 Inferential Analytics Assignment

## Objective

In this assignment, you will use NHANES data to perform basic inferential statistics using Python in Google Colab. You will explore relationships and differences in health metrics and demographic variables, utilizing the skills learned in class to answer key questions about the dataset. Your final analysis should be saved as a Google Colab notebook and uploaded to a GitHub repository.

* NHANES Data: [NHANES 2021-2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)

## Data Preparation

To start, you’ll use the following NHANES variables for analysis:

* **Marital Status** (`DMDMARTZ`) - categorical, needs recoding (married or not married).
* **Education Level** (`DMDEDUC2`) - categorical, needs recoding (bachelor’s or higher vs. less than bachelor’s).
* **Age in Years** (`RIDAGEYR`) - continuous.
* **Systolic Blood Pressure** (`BPXOSY3`) - continuous.
* **Diastolic Blood Pressure** (`BPXODI3`) - continuous.
* **Vitamin D Lab Interpretation** (`LBDVD2LC`) - categorical, two levels.
* **Hepatitis B Lab Antibodies** (`LBXHBS`) - categorical, needs recoding to two levels.
* **Weak/Failing Kidneys** (`KIQ022`) - categorical, can be treated as two levels.
* **Minutes of Sedentary Behavior** (`PAD680`) - continuous, needs cleaning (remove values `7777`, `9999`, and null).
* **Current Self-Reported Weight** (`WHD020`) - continuous, needs cleaning (remove values `7777`, `9999`, and null).

> **Note** : Ensure you clean the data before performing analyses. For  **categorical variables** , check and document frequency counts to confirm data consistency. For  **continuous variables** , be mindful of placeholder values (`7777`, `9999`) and handle these as appropriate (e.g., by removing or imputing).

## Instructions

[](https://github.com/hantswilliams/HHA-507-2025/blob/main/assignments/assignment4_files/assignment_inferential.md#instructions)

1. **Create Your GitHub Repository**
   * Create a new GitHub repository titled `nhanes_inferential_2021_23`.
   * Include a `README.md` file that briefly describes the project and the analyses you are performing for each of the questions.
2. **Complete the Analysis in Google Colab**
   * Use Google Colab to conduct your analysis. Your notebook should be well-documented with explanations of each step just as like we have done in the class notebooks.
3. **Questions for Analysis**
   Use the questions below to guide your analysis. Remember to transform or recode variables where needed as specified, and determine the appropriate statistical tests that should be performed based on the question and variables.
   * **Question 1** : "Is there an association between marital status (married or not married) and education level (bachelor’s degree or higher vs. less than a bachelor’s degree)?"
   * Variables: `DMDMARTZ` (marital status) and `DMDEDUC2` (education level). Recode as specified.
   * **Question 2** : "Is there a difference in the mean sedentary behavior time between those who are married and those who are not married?"
   * Variables: `DMDMARTZ` (marital status, recoded) and `PAD680` (sedentary behavior time, cleaned).
   * **Question 3** : "How do age and marital status affect systolic blood pressure?"
   * Variables: `RIDAGEYR` (age), `DMDMARTZ` (marital status, recoded), and `BPXOSY3` (systolic blood pressure).
   * **Question 4** : "Is there a correlation between self-reported weight and minutes of sedentary behavior?"
   * Variables: `WHD020` (self-reported weight, cleaned) and `PAD680` (sedentary behavior time, cleaned).
   * **Question 5 (Creative Analysis)** : Develop your own unique question using at least one of the variables listed above. Ensure that your question can be answered using one of the following tests: chi-square, t-test, ANOVA, or correlation. Clearly state your question, explain why you chose the test, and document your findings.
4. **Deliverables**
   * A completed Google Colab notebook (`.ipynb` file) with:
     * Data loading and preparation steps
     * Analysis and any transformations
     * Visualizations of descriptives/results (if relevant)
     * Brief summaries of your findings for each question
   * Submit the link to your GitHub repository titled `nhanes_inferential_2023`.

---

Ensure your repository is public or accessible by link, and confirm that all code, results, and any explanations are documented clearly within your notebook.

If you have any issues or run into errors, please be sure to screen shot the error message and include it in your notebook. This will help me understand the problem and provide guidance on how to resolve it. Do not NOT submit because of errors.

Initially

Used extension SAS viewer to open xpt files in vscode and exported it as xlsx files

Then tried using openpyxl inorder to read xlsx files and run the code by trial and error

However only 216 rows were showing

demo = pd.ExcelFile('data/DEMO_L_20251021T161012.xlsx').parse('Records', header=0)

Then decided to directly use xpt files



## **2.1 Marital Status (DMDMARTZ)**

**Codebook values:**

* 1 = Married/Living with partner (Count: 4136)
* 2 = Widowed/Divorced/Separated (Count: 2022)
* 3 = Never married (Count: 1625)
* 77 = Refused (Count: 4)
* 99 = Don't know (Count: 5)
* Missing (Count: 4141)



## **2.2 Education Level (DMDEDUC2)**

**Codebook values:**

* 1 = Less than 9th grade (Count: 373)
* 2 = 9-11th grade (Count: 666)
* 3 = High school graduate/GED (Count: 1749)
* 4 = Some college or AA degree (Count: 2370)
* 5 = College graduate or above (Count: 2625)
* 7 = Refused (Count: 0)
* 9 = Don't know (Count: 11)
* Missing (Count: 4139)



## **2.3 Kidney Health (KIQ022)**

**Codebook values:**

* 1 = Yes (Count: 321)
* 2 = No (Count: 7473)
* 7 = Refused (Count: 0)
* 9 = Don't know (Count: 13)
* Missing (Count: 2)



## **2.4 Sedentary Minutes (PAD680)**

**Codebook values:**

* 0 to 1380 = Range of Values (Count: 8065)
* 7777 = Refused (Count: 6)
* 9999 = Don't know (Count: 67)
* Missing (Count: 15)



## **2.5 Self-Reported Weight (WHD020)**

**Codebook values:**

* 63 to 530 = Range of Values (Count: 8358)
* 7777 = Refused (Count: 40)
* 9999 = Don't know (Count: 88)
* Missing (Count: 15)



## **2.6 Vitamin D Lab (LBDVD2LC)**

**Codebook values:**

* 0 = At or above the detection limit (Count: 1447)
* 1 = Below lower detection limit (Count: 5860)
* Missing (Count: 1420)


## **2.7 Hepatitis B Surface Antibody (LBXHBS)**

**Codebook values:**

* 1 = Positive (Count: 2042)
* 2 = Negative (Count: 5324)
* 3 = Indeterminate (Count: 0)
* Missing (Count: 1245)


## **2.7 Hepatitis B Surface Antibody (LBXHBS)**

**Codebook values:**

* 1 = Positive (Count: 2042)
* 2 = Negative (Count: 5324)
* 3 = Indeterminate (Count: 0)
* Missing (Count: 1245)

## **2.8 Blood Pressure (BPXOSY3, BPXODI3)**

**Codebook values:**

* BPXOSY3: 50 to 232 mmHg (Count: 7480), Missing (321)
* BPXODI3: 24 to 136 mmHg (Count: 7480), Missing (321)


## **2.9 Age (RIDAGEYR)**

**Codebook values:**

* 0 to 79 = Range of Values (Count: 11408)
* 80 = 80 years of age and over (Count: 525)
* Missing (Count: 0)



## **Step 3: Merge All DataFrames by SEQN**



## **✅Current Dataset Summary**

| Variable           | Description          | Type        | Valid Values                          |
| ------------------ | -------------------- | ----------- | ------------------------------------- |
| SEQN               | Respondent ID        | ID          | All 11,933                            |
| RIDAGEYR           | Age in years         | Continuous  | 0-80                                  |
| RIAGENDR           | Gender               | Categorical | 1=Male, 2=Female                      |
| married            | Marital status       | Binary      | 1=Married, 0=Not married, NaN=Unknown |
| bachelor_or_higher | Education            | Binary      | 1=Bachelor's+, 0=Less, NaN=Unknown    |
| weak_kidney        | Kidney health        | Binary      | 1=Yes, 0=No, NaN=Unknown              |
| BPXOSY3            | Systolic BP          | Continuous  | 50-232 mmHg                           |
| BPXODI3            | Diastolic BP         | Continuous  | 24-136 mmHg                           |
| PAD680             | Sedentary minutes    | Continuous  | 0-1380 min                            |
| WHD020             | Weight (pounds)      | Continuous  | 63-530 lbs                            |
| vitamin_d_status   | Vitamin D lab        | Binary      | 0=At/above limit, 1=Below limit       |
| hepb_positive      | Hepatitis B antibody | Binary      | 1=Positive, 0=Negative, NaN=Unknown   |
