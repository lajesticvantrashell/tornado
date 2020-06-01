import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from numpy.random import randint
from streams.generators.tools.transition_functions import Transition

class MIMIC:

    '''
    When creating a synthetic MIMIC-based dataset, this object
    contains a synthetic concept.
    '''

    MIMIC_DATA = None
    PATH_TO_DATA = None

    @staticmethod
    def set_path_to_data(path):
        MIMIC.PATH_TO_DATA = path

    @staticmethod
    def get_mimic_data():
        # Load MIMIC data if it hasn't been already.
        if type(MIMIC.MIMIC_DATA) is type(None):
            MIMIC.MIMIC_DATA = pd.read_csv(MIMIC.PATH_TO_DATA)
        return MIMIC.MIMIC_DATA

    def __init__(self, concept_length=1000, transition_length=50,
        noise_rate=0.1, n_priorities=4, n_concepts=2, random_seed=None):

        if random_seed==None:
            random_seed=random.randint(1000000)

        self.__INSTANCES_NUM = concept_length * n_concepts
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = n_concepts - 1
        self.__W = transition_length
        self.__RECORDS = []
        self.__N_PRIORITIES = n_priorities
        self.__N_CONCEPTS = n_concepts

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__INSTANCES_NUM), int(self.__INSTANCES_NUM * noise_rate))

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    @staticmethod
    def get_class_name():
        return 'MIMIC'

    def generate_concept(self, n_random_labels=20):
        concept = DecisionTreeClassifier()
        rand_labels = randint(1, self.__N_PRIORITIES+1, size=n_random_labels)
        concept.fit(self.__FEATURES_DF.iloc[:n_random_labels, :], rand_labels)
        # print(self.__FEATURES_DF.iloc[:n_random_labels, :])
        return concept

    def generate(self, output_path="MIMIC"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING CONCEPTS
        features_df = self.__FEATURES_DF = MIMIC.get_mimic_data().sample(self.__INSTANCES_NUM)
        concepts = self.__CONCEPTS = [ self.generate_concept() for i in range(self.__N_CONCEPTS) ]

        # [2] CREATING RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            context_id = int(i / self.__CONCEPT_LENGTH)
            record = self.create_record(i, context_id)
            self.__RECORDS.append(list(record))

        # [3] TRANSITION
        for i in range(1, self.__NUM_DRIFTS + 1):
            transition = []
            for j in range(0, self.__W):
                instance_index = i * self.__CONCEPT_LENGTH + j
                if random.random() < Transition.sigmoid(j, self.__W):
                    # concept = self.__CONCEPTS[i-1]
                    context_id = i-1
                else:
                    # concept = self.__CONCEPTS[i]
                    context_id = i
                record = self.create_record(instance_index, context_id)
                transition.append(list(record))
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            self.__RECORDS[starting_index: ending_index] = transition

        # [4] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, i, dist_id):
        # i is the index of the instance in the features_df
        # dist_id is the concept index
        features = self.__FEATURES_DF.iloc[i, :]
        concept = self.__CONCEPTS[dist_id]
        # print(features)
        # print(features.to_numpy().reshape(1, 1))
        # print(features.to_numpy().reshape(-1, 1).shape())
        label = concept.predict(features.to_numpy().reshape(1, -1))
        return list(features) + [label]

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            c = self.__RECORDS[noise_spot][2]
            rand_add = random.randint(1, self.__N_PRIORITIES)
            self.__RECORDS[noise_spot][2] = (c+rand_add)%5

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation MIMIC\n")
        for col in self.__FEATURES_DF.columns:
            arff_writer.write(f"@attribute {col} real \n")
        classes_string = ','.join(str(i+1) for i in range(self.__N_PRIORITIES))
        arff_writer.write(f"@attribute priority {classes_string} real \n\n")
        arff_writer.write("@data" + "\n")
        for record in self.__RECORDS:
            line_str = ",".join(str(i) for i in record)
            arff_writer.write(line_str + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")


# class MIMIC_NEGATE(MIMIC):
#
#     def __init__(self, concept_length=20000, transition_length=50,
#         noise_rate=0.1, n_priorities=4, negate_n=None, n_concepts=5, random_seed=None)
#
#         self.negate_n = negate_n # default conditino 10% of features
#
#         super().__init__(concept_length=20000, transition_length=50,
#             noise_rate=0.1, n_priorities=4, n_concepts=5, random_seed=None)
#
#     def generate(self):
#         pass
#
# class MIMIC_UNDERSAMPLE(MIMIC):
#     def __init__(self):
#         pass
#
# class MIMIC_OVERSAMPLE(MIMIC):
#     def __init__(self):
#         pass
