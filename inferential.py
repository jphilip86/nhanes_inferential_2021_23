
import pandas as pd
import numpy as np
# import openpyxl  # Thought I will convert xpt to xlsx .openpyxl was needed for reading .xlsx files with pandas however only showed 216 rows instead of 11,933. So then decided to directly use sas files with pd.read_sas.

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Step 1: Load NHANES datasets

# Load Demographics - 11,933 records
demo = pd.read_sas('data/DEMO_L.XPT', format='xport')
print(f"Demographics: {demo.shape}")

# Load Kidney Health
kiq = pd.read_sas('data/KIQ_U_L.XPT', format='xport')
print(f"Kidney: {kiq.shape}")

# Load Blood Pressure
bpxo = pd.read_sas('data/BPXO_L.XPT', format='xport')
print(f"Blood Pressure: {bpxo.shape}")

# Load Physical Activity
paq = pd.read_sas('data/PAQ_L.XPT', format='xport')
print(f"Physical Activity: {paq.shape}")

# Load Weight/Height Questionnaire
whq = pd.read_sas('data/WHQ_L.XPT', format='xport')
print(f"Weight: {whq.shape}")

# Load Vitamin D Lab
vid = pd.read_sas('data/VID_L.XPT', format='xport')
print(f"Vitamin D: {vid.shape}")

# Load Hepatitis B Serology
hepb = pd.read_sas('data/HEPB_S_L.XPT', format='xport')
print(f"Hepatitis B: {hepb.shape}")




# Step 2: Clean and recode variables 

# 2.1 Marital Status → married (binary) 2.1 Marital Status (DMDMARTZ)
# Codebook values:
# 1 = Married/Living with partner (Count: 4136)
# 2 = Widowed/Divorced/Separated (Count: 2022)
# 3 = Never married (Count: 1625)
# 77 = Refused (Count: 4)
# 99 = Don't know (Count: 5)
# Missing (Count: 4141)

def recode_marital(x):
    """
    NHANES DMDMARTZ codes:
    1 = Married/Living with partner → 1
    2 = Widowed/Divorced/Separated → 0
    3 = Never married → 0
    77, 99, NaN = Refused/Don't know/Missing → NaN
    """
    if x == 1:
        return 1
    elif x in [2, 3]:
        return 0
    elif x in [77, 99] or pd.isna(x):
        return np.nan
    else:
        return np.nan

demo['married'] = demo['DMDMARTZ'].apply(recode_marital)
print("Marital Status Recoding:")
print(demo['married'].value_counts(dropna=False))



# 2.2 Education Level → bachelor_or_higher 2.2 Education Level (DMDEDUC2)
# Codebook values:
# 1 = Less than 9th grade (Count: 373)
# 2 = 9-11th grade (Count: 666)
# 3 = High school graduate/GED (Count: 1749)
# 4 = Some college or AA degree (Count: 2370)
# 5 = College graduate or above (Count: 2625)
# 7 = Refused (Count: 0)
# 9 = Don't know (Count: 11)
# Missing (Count: 4139)

def recode_education(x):
    """
    5 = College graduate or above → 1 (bachelor's+)
    1-4 = Less than bachelor's → 0
    7, 9, NaN = Unknown → NaN
    """
    if x == 5:
        return 1
    elif x in [1, 2, 3, 4]:
        return 0
    elif x in [7, 9] or pd.isna(x):
        return np.nan
    else:
        return np.nan

demo['bachelor_or_higher'] = demo['DMDEDUC2'].apply(recode_education)
print("\nEducation Level Recoding:")
print(demo['bachelor_or_higher'].value_counts(dropna=False))


# 2.3 Kidney Health (KIQ022)
# Codebook values:
# 1 = Yes (Count: 321)
# 2 = No (Count: 7473)
# 7 = Refused (Count: 0)
# 9 = Don't know (Count: 13)
# Missing (Count: 2)

def recode_kidney(x):
    """
    1 = Yes, told had weak/failing kidneys → 1
    2 = No → 0
    7, 9, NaN = Unknown → NaN
    """
    if x == 1:
        return 1
    elif x == 2:
        return 0
    elif x in [7, 9] or pd.isna(x):
        return np.nan
    else:
        return np.nan

kiq['weak_kidney'] = kiq['KIQ022'].apply(recode_kidney)
print("\nKidney Health Recoding:")
print(kiq['weak_kidney'].value_counts(dropna=False))


# 2.4 Sedentary Minutes (PAD680)Codebook values:
# 0 to 1380 = Range of Values (Count: 8065)
# 7777 = Refused (Count: 6)
# 9999 = Don't know (Count: 67)
# Missing (Count: 15)
# Replace 7777, 9999 with NaN


paq['PAD680'] = paq['PAD680'].replace([7777, 9999], np.nan)
print("\nSedentary Minutes Cleaning:")
print(f"Valid values: {paq['PAD680'].notna().sum()}")
print(f"Missing: {paq['PAD680'].isna().sum()}")



# 2.5 Self-Reported Weight (WHD020)
# Codebook values:
# 63 to 530 = Range of Values (Count: 8358)
# 7777 = Refused (Count: 40)
# 9999 = Don't know (Count: 88)
# Missing (Count: 15)
# Replace 7777, 9999 with NaN

whq['WHD020'] = whq['WHD020'].replace([7777, 9999], np.nan)
print("\nWeight Cleaning:")
print(f"Valid values: {whq['WHD020'].notna().sum()}")
print(f"Missing: {whq['WHD020'].isna().sum()}")



# 2.6 Vitamin D Lab (LBDVD2LC)
# Codebook values:
# 0 = At or above the detection limit (Count: 1447)
# 1 = Below lower detection limit (Count: 5860)
# Missing (Count: 1420)
# Keep as-is, it's already binary


vid['vitamin_d_status'] = vid['LBDVD2LC']
print("\nVitamin D Lab Status:")
print(vid['vitamin_d_status'].value_counts(dropna=False))



# 2.7 Hepatitis B Surface Antibody (LBXHBS)
# Codebook values:
# 1 = Positive (Count: 2042)
# 2 = Negative (Count: 5324)
# 3 = Indeterminate (Count: 0)
# Missing (Count: 1245)


def recode_hepb(x):
    """
    1 = Positive → 1
    2 = Negative → 0
    3, NaN = Indeterminate/Missing → NaN
    """
    if x == 1:
        return 1
    elif x == 2:
        return 0
    elif x == 3 or pd.isna(x):
        return np.nan
    else:
        return np.nan

hepb['hepb_positive'] = hepb['LBXHBS'].apply(recode_hepb)
print("\nHepatitis B Antibody Recoding:")
print(hepb['hepb_positive'].value_counts(dropna=False))


# 2.8 Blood Pressure (BPXOSY3, BPXODI3)
# Codebook values:
# BPXOSY3: 50 to 232 mmHg (Count: 7480), Missing (321)
# BPXODI3: 24 to 136 mmHg (Count: 7480), Missing (321)
# No cleaning needed - continuous values are valid


print("\nBlood Pressure - Systolic:")
print(bpxo['BPXOSY3'].describe())
print("\nBlood Pressure - Diastolic:")
print(bpxo['BPXODI3'].describe())



# 2.9 Age (RIDAGEYR)
# Codebook values:
# 0 to 79 = Range of Values (Count: 11408)
# 80 = 80 years of age and over (Count: 525)
# Missing (Count: 0)
# No recoding needed - age is continuous

print("\nAge Distribution:")
print(demo['RIDAGEYR'].describe())





# Step 3: Merge All DataFrames by SEQN

# Start with demographics
merged = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'married', 'bachelor_or_higher']].copy()

# Merge kidney
merged = merged.merge(kiq[['SEQN', 'weak_kidney']], on='SEQN', how='left')

# Merge blood pressure
merged = merged.merge(bpxo[['SEQN', 'BPXOSY3', 'BPXODI3']], on='SEQN', how='left')

# Merge physical activity
merged = merged.merge(paq[['SEQN', 'PAD680']], on='SEQN', how='left')

# Merge weight
merged = merged.merge(whq[['SEQN', 'WHD020']], on='SEQN', how='left')

# Merge vitamin D
merged = merged.merge(vid[['SEQN', 'vitamin_d_status']], on='SEQN', how='left')

# Merge hepatitis B
merged = merged.merge(hepb[['SEQN', 'hepb_positive']], on='SEQN', how='left')

print(f"\n✓ Final merged dataset shape: {merged.shape}")
print(merged.head())


# Step 4: Data Integrity Checks

print("\n=== Missing Values Summary ===")
print(merged.isnull().sum())
print(f"\nTotal records: {len(merged)}")

print("\n=== Categorical Variable Frequencies ===")
for var in ['married', 'bachelor_or_higher', 'weak_kidney', 'vitamin_d_status', 'hepb_positive']:
    print(f"\n{var}:")
    print(merged[var].value_counts(dropna=False))

print("\n=== Continuous Variable Descriptives ===")
print(merged[['RIDAGEYR', 'PAD680', 'WHD020', 'BPXOSY3', 'BPXODI3']].describe())

# Question 1: Association Between Marital Status and Education Level
# Research Question: Is there an association between marital status (married or not married) and education level (bachelor's degree or higher vs. less than a bachelor's degree)?
# Variables: married (recoded from DMDMARTZ) and bachelor_or_higher (recoded from DMDEDUC2)
# Appropriate Test: Chi-Square Test of Independence (both variables are categorical)

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("QUESTION 1: Association Between Marital Status and Education Level")
print("="*70)

# Remove missing values
q1_data = merged[['married', 'bachelor_or_higher']].dropna()

# Create contingency table
crosstab = pd.crosstab(q1_data['married'], q1_data['bachelor_or_higher'], margins=True)
print("\nContingency Table (Marital Status vs Education):")
print(crosstab)

# Perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(
    pd.crosstab(q1_data['married'], q1_data['bachelor_or_higher'])
)

print(f"\nChi-Square Test Results:")
print(f"  Chi-square statistic: {chi2:.3f}")
print(f"  p-value: {p:.4f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Sample size: {len(q1_data)}")

# Interpretation
if p < 0.05:
    print(f"\n✓ SIGNIFICANT: There IS a statistically significant association")
    print(f"  between marital status and education level (p < 0.05)")
else:
    print(f"\n✗ NOT SIGNIFICANT: There is NO statistically significant association")
    print(f"  between marital status and education level (p ≥ 0.05)")

# Visualization
plt.figure(figsize=(10, 6))
ct = pd.crosstab(q1_data['married'], q1_data['bachelor_or_higher'], normalize='index') * 100
ct.plot(kind='bar', color=['#e74c3c', '#3498db'])
plt.title("Education Level by Marital Status", fontsize=14, fontweight='bold')
plt.xlabel("Marital Status (0=Not Married, 1=Married)")
plt.ylabel("Percentage (%)")
plt.legend(["< Bachelor's", "Bachelor's+"], title="Education")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# 5.2 Question 2: T-Test - Sedentary Behavior by Marital Status
# Research Question: Do married individuals differ from not married individuals in their average daily sedentary time (in minutes)?

print("\n" + "="*70)
print("QUESTION 2: Sedentary Behavior by Marital Status")
print("="*70)

# Remove missing values
q2_data = merged[['married', 'PAD680']].dropna()

# Separate groups
married_group = q2_data[q2_data['married'] == 1]['PAD680']
not_married_group = q2_data[q2_data['married'] == 0]['PAD680']

# Descriptive statistics
print(f"\nDescriptive Statistics:")
print(f"  Married: n={len(married_group)}, Mean={married_group.mean():.2f} min, SD={married_group.std():.2f}")
print(f"  Not Married: n={len(not_married_group)}, Mean={not_married_group.mean():.2f} min, SD={not_married_group.std():.2f}")
print(f"  Mean difference: {married_group.mean() - not_married_group.mean():.2f} minutes")

# Independent samples t-test
t_stat, p_val = stats.ttest_ind(married_group, not_married_group)

print(f"\nIndependent T-Test Results:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.4f}")

# Interpretation
if p_val < 0.05:
    print(f"\n✓ SIGNIFICANT: There IS a statistically significant difference in")
    print(f"  sedentary time between married and not married groups (p < 0.05)")
else:
    print(f"\n✗ NOT SIGNIFICANT: There is NO statistically significant difference in")
    print(f"  sedentary time between married and not married groups (p ≥ 0.05)")



# 5.3 Question 3: Multiple Regression - Age and Marital Status Effect on Systolic BP

# Add these imports to your script
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("\n" + "="*70)
print("QUESTION 3: Age and Marital Status Effects on Systolic BP")
print("="*70)

# Remove missing values
q3_data = merged[['RIDAGEYR', 'married', 'BPXOSY3']].dropna()

# Prepare data for regression
X = q3_data[['RIDAGEYR', 'married']]
y = q3_data['BPXOSY3']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Results
print(f"\nMultiple Linear Regression Results:")
print(f"  Sample size: {len(q3_data)}")
print(f"  R-squared: {r2_score(y, y_pred):.4f}")
print(f"\nRegression Coefficients:")
print(f"  Intercept: {model.intercept_:.3f} mmHg")
print(f"  Age coefficient: {model.coef_[0]:.3f} mmHg per year")
print(f"  Marital status coefficient: {model.coef_[1]:.3f} mmHg")

# Individual correlations
r_age, p_age = stats.pearsonr(q3_data['RIDAGEYR'], q3_data['BPXOSY3'])
print(f"\nAge-BP Correlation:")
print(f"  Pearson r: {r_age:.3f}, p-value: {p_age:.4f}")

# Marital status effect (t-test)
married_bp = q3_data[q3_data['married'] == 1]['BPXOSY3']
not_married_bp = q3_data[q3_data['married'] == 0]['BPXOSY3']
t_stat, p_marital = stats.ttest_ind(married_bp, not_married_bp)

print(f"\nMarital Status Effect on BP:")
print(f"  Married mean: {married_bp.mean():.2f} mmHg (n={len(married_bp)})")
print(f"  Not married mean: {not_married_bp.mean():.2f} mmHg (n={len(not_married_bp)})")
print(f"  t-statistic: {t_stat:.3f}, p-value: {p_marital:.4f}")

# Interpretation
print(f"\n📊 INTERPRETATION:")
if p_age < 0.05:
    print(f"  ✓ Age significantly affects systolic BP (p < 0.05)")
else:
    print(f"  ✗ Age does NOT significantly affect systolic BP (p ≥ 0.05)")
    
if p_marital < 0.05:
    print(f"  ✓ Marital status significantly affects systolic BP (p < 0.05)")
else:
    print(f"  ✗ Marital status does NOT significantly affect systolic BP (p ≥ 0.05)")


# 5.4 Question 4: Pearson Correlation - Weight vs Sedentary Behavior

print("\n" + "="*70)
print("QUESTION 4: Correlation Between Weight and Sedentary Behavior")
print("="*70)

# Remove missing values
q4_data = merged[['WHD020', 'PAD680']].dropna()

# Calculate Pearson correlation
r, p_val = stats.pearsonr(q4_data['WHD020'], q4_data['PAD680'])

print(f"\nPearson Correlation Results:")
print(f"  Sample size: {len(q4_data)}")
print(f"  Pearson r: {r:.3f}")
print(f"  p-value: {p_val:.4f}")
print(f"  R-squared: {r**2:.4f} ({r**2*100:.2f}% of variance explained)")

# Descriptive statistics
print(f"\nDescriptive Statistics:")
print(f"  Weight: Mean={q4_data['WHD020'].mean():.2f} lbs, SD={q4_data['WHD020'].std():.2f}")
print(f"  Sedentary time: Mean={q4_data['PAD680'].mean():.2f} min, SD={q4_data['PAD680'].std():.2f}")

# Interpretation
print(f"\n📊 INTERPRETATION:")
if p_val < 0.05:
    if abs(r) < 0.3:
        strength = "weak"
    elif abs(r) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if r > 0 else "negative"
    
    print(f"  ✓ SIGNIFICANT: There IS a statistically significant {strength} {direction}")
    print(f"    correlation between weight and sedentary behavior (p < 0.05)")
else:
    print(f"  ✗ NOT SIGNIFICANT: There is NO statistically significant correlation")
    print(f"    between weight and sedentary behavior (p ≥ 0.05)")


# 5.5 Question 5: Creative Analysis - Vitamin D Status and Blood Pressure

print("\n" + "="*70)
print("QUESTION 5: CREATIVE ANALYSIS")
print("Vitamin D Status and Systolic Blood Pressure")
print("="*70)

print("\n📋 RESEARCH QUESTION:")
print("Is there a difference in systolic blood pressure between individuals")
print("with adequate vitamin D levels vs. those with low vitamin D levels?")

print("\n🔬 TEST CHOICE: Independent Samples T-Test")
print("RATIONALE: Comparing mean systolic BP between two independent groups")
print("(adequate vitamin D vs. low vitamin D)")

# Remove missing values
q5_data = merged[['vitamin_d_status', 'BPXOSY3']].dropna()

# Separate groups
# 0 = At or above detection limit (adequate)
# 1 = Below detection limit (low)
adequate_vit_d = q5_data[q5_data['vitamin_d_status'] == 0]['BPXOSY3']
low_vit_d = q5_data[q5_data['vitamin_d_status'] == 1]['BPXOSY3']

# Descriptive statistics
print(f"\nDescriptive Statistics:")
print(f"  Adequate Vitamin D: n={len(adequate_vit_d)}, Mean={adequate_vit_d.mean():.2f} mmHg, SD={adequate_vit_d.std():.2f}")
print(f"  Low Vitamin D: n={len(low_vit_d)}, Mean={low_vit_d.mean():.2f} mmHg, SD={low_vit_d.std():.2f}")
print(f"  Mean difference: {low_vit_d.mean() - adequate_vit_d.mean():.2f} mmHg")

# Independent samples t-test
t_stat, p_val = stats.ttest_ind(adequate_vit_d, low_vit_d)

print(f"\nIndependent T-Test Results:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.4f}")

# Interpretation
print(f"\n📊 FINDINGS:")
if p_val < 0.05:
    print(f"  ✓ SIGNIFICANT: There IS a statistically significant difference in")
    print(f"    systolic BP between vitamin D groups (p < 0.05)")
    if low_vit_d.mean() > adequate_vit_d.mean():
        print(f"    Individuals with LOW vitamin D have HIGHER blood pressure.")
    else:
        print(f"    Individuals with ADEQUATE vitamin D have HIGHER blood pressure.")
else:
    print(f"  ✗ NOT SIGNIFICANT: There is NO statistically significant difference in")
    print(f"    systolic BP between vitamin D groups (p ≥ 0.05)")

print("\n💡 CLINICAL RELEVANCE:")
print("This analysis explores the relationship between vitamin D status and")
print("cardiovascular health, as vitamin D deficiency has been linked to")
print("increased cardiovascular disease risk in epidemiological studies.")



# Step 6: Visualizations for All 5 Questions
# Create a figure with all 5 visualizations


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Create a figure with all 5 visualizations
fig = plt.figure(figsize=(16, 12))

# ==========================================
# 6.1 Question 1: Marital Status vs Education
# ==========================================
plt.subplot(3, 2, 1)
q1_plot = pd.crosstab(merged['married'], merged['bachelor_or_higher'], normalize='index') * 100
q1_plot.plot(kind='bar', ax=plt.gca(), color=['#e74c3c', '#3498db'])
plt.title("Q1: Education Level by Marital Status", fontweight='bold', fontsize=12)
plt.xlabel("Marital Status (0=Not Married, 1=Married)")
plt.ylabel("Percentage (%)")
plt.legend(["< Bachelor's", "Bachelor's+"], title="Education")
plt.xticks(rotation=0)

# ==========================================
# 6.2 Question 2: Sedentary Minutes by Marital Status
# ==========================================
plt.subplot(3, 2, 2)
q2_plot = merged[['married', 'PAD680']].dropna()
q2_plot['Status'] = q2_plot['married'].map({0: 'Not Married', 1: 'Married'})
sns.boxplot(x='Status', y='PAD680', data=q2_plot, palette=['#e74c3c', '#3498db'], ax=plt.gca())
plt.title("Q2: Sedentary Minutes by Marital Status", fontweight='bold', fontsize=12)
plt.ylabel("Sedentary Minutes per Day")
plt.xlabel("Marital Status")

# ==========================================
# 6.3 Question 3a: Age vs Systolic BP
# ==========================================
plt.subplot(3, 2, 3)
q3_plot = merged[['RIDAGEYR', 'BPXOSY3']].dropna()
plt.scatter(q3_plot['RIDAGEYR'], q3_plot['BPXOSY3'], alpha=0.3, s=10, color='#3498db')
# Add regression line
z = np.polyfit(q3_plot['RIDAGEYR'], q3_plot['BPXOSY3'], 1)
p = np.poly1d(z)
plt.plot(q3_plot['RIDAGEYR'], p(q3_plot['RIDAGEYR']), "r-", linewidth=2)
plt.xlabel("Age (years)")
plt.ylabel("Systolic BP (mmHg)")
plt.title("Q3a: Age vs Systolic Blood Pressure", fontweight='bold', fontsize=12)

# ==========================================
# 6.4 Question 3b: Systolic BP by Marital Status
# ==========================================
plt.subplot(3, 2, 4)
q3b_plot = merged[['married', 'BPXOSY3']].dropna()
q3b_plot['Status'] = q3b_plot['married'].map({0: 'Not Married', 1: 'Married'})
sns.boxplot(x='Status', y='BPXOSY3', data=q3b_plot, palette=['#e74c3c', '#3498db'], ax=plt.gca())
plt.title("Q3b: Systolic BP by Marital Status", fontweight='bold', fontsize=12)
plt.ylabel("Systolic BP (mmHg)")
plt.xlabel("Marital Status")

# ==========================================
# 6.5 Question 4: Weight vs Sedentary Behavior
# ==========================================
plt.subplot(3, 2, 5)
q4_plot = merged[['WHD020', 'PAD680']].dropna()
plt.scatter(q4_plot['WHD020'], q4_plot['PAD680'], alpha=0.3, s=10, color='#2ecc71')
# Add regression line
z = np.polyfit(q4_plot['WHD020'], q4_plot['PAD680'], 1)
p = np.poly1d(z)
plt.plot(q4_plot['WHD020'], p(q4_plot['WHD020']), "r-", linewidth=2)
plt.xlabel("Self-Reported Weight (pounds)")
plt.ylabel("Sedentary Minutes per Day")
plt.title("Q4: Weight vs Sedentary Behavior", fontweight='bold', fontsize=12)

# ==========================================
# 6.6 Question 5: Vitamin D Status vs Systolic BP
# ==========================================
plt.subplot(3, 2, 6)
q5_plot = merged[['vitamin_d_status', 'BPXOSY3']].dropna()
q5_plot['Vitamin D'] = q5_plot['vitamin_d_status'].map({0: 'Adequate', 1: 'Low'})
sns.boxplot(x='Vitamin D', y='BPXOSY3', data=q5_plot, palette=['#2ecc71', '#e67e22'], ax=plt.gca())
plt.title("Q5: Systolic BP by Vitamin D Status", fontweight='bold', fontsize=12)
plt.ylabel("Systolic BP (mmHg)")
plt.xlabel("Vitamin D Status")

plt.tight_layout()
plt.savefig('nhanes_all_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ All visualizations created and saved as 'nhanes_all_visualizations.png'")
