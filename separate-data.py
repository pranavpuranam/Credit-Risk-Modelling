from preprocessing import *

all_data = pd.read_csv("standard.csv")

# split data into training and testing sets (80% for training and 20% for testing)

train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)

# save all to csv file format

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
