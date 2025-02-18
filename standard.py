from preprocessing import *
from sklearn.preprocessing import StandardScaler

# import the excel files

nonstandard = pd.read_csv("num-data.csv")

print(nonstandard.head())

# turn all the bools into 1s and 0s

bool_cols = nonstandard.select_dtypes(include=['bool']).columns

for col in bool_cols:
    nonstandard[col] =nonstandard[col].astype(int)

print(nonstandard.info())

# standardise all values

num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_grade','loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

scaler = StandardScaler()

nonstandard[num_cols] = scaler.fit_transform(raw[num_cols])

# Save standardized dataset
nonstandard.to_csv('standard.csv', index=False)




"""
# split data into training and testing sets (80% for training and 20% for testing)

train_df, test_df = train_test_split(raw, test_size=0.2, random_state=42)

# save all to csv file format

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
"""