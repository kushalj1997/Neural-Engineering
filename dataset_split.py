# BIOMEDE 517 - Neural Engineering
# Neural Networks for Neural Networks - Understanding Seizure EEG Data
# This file is used to create training and test datasets

def clean_string(s):
    # s = s.replace("\\n", "")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("\"", "")
    return s

def main():
    # Grab the og file
    eeg_data = open("Final Project Data/SpottingSeizure/data.csv", "r")
    # Create the new training and test files
    eeg_train = open("Final Project Data/SpottingSeizure/eeg_train.csv", "w")
    eeg_test = open("Final Project Data/SpottingSeizure/eeg_test.csv", "w")

    # Number of examples in the training data (11500 rows x 180 samples)
    num_total_examples = 11500
    samples_per_example = 180

    # Remove the header from the data
    header = eeg_data.readline()

    # Original dataset contains 11500 samples, put half of these in
    # training data and other half in testing data
    train_samples = ""
    test_samples = ""
    num_lines = 0
    for line in eeg_data:
        # Take out junk values from data file
        line = clean_string(line)
        if num_lines < num_total_examples/2:
            test_samples = test_samples + str(line)
        else:
            train_samples = train_samples + str(line)
        num_lines += 1

    print(train_samples)

    eeg_train.write(train_samples)
    eeg_test.write(test_samples)

    print("Number of training samples: {}".format(len(train_samples)))
    print("Example Training Sample: {}".format(train_samples[1]))
    print("Number of testing samples: {}".format(len(train_samples)))
    print("Example Testing Sample: {}".format(test_samples[1]))

    eeg_data.close()
    eeg_train.close()
    eeg_test.close()

if __name__ == '__main__':
    main()
