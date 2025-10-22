
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

