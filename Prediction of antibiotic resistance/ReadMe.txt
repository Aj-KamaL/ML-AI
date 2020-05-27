AIM: Antibiotic Resistance Prediction for E.Coli Strains. 

Features Used: ["GS", "G", "S", "GY", "GYS", "SY"] where 
G: gene information
S: Population Structure
Y: Year of Isolation of the Strains and other Metadata

Antibiotics(Drugs) Used along with their Drug Code:
CTX(1)
AMP(2)
AMX(3)
AMC(4)
CTZ(5)
CXM(6)
CET(7)
GEN(8)
TBM(9)
TMP(10)
CIP(11)


Running the Code(from the Project's Root Directory: inside PanPred):

1) First, create the feature file for a given drug for prediction using the Metadata, Population Structure and Gene Information.

	python PanPred.py -c "create" -i "GY" -d "1" -m "./test_data/Metadata.csv" -g "./test_data/AccessoryGene.csv" -s "./test_data/PopulationStructure.csv_labelencoded.csv"

	python PanPred.py -c "create" -i "G" -d "1" -m "./test_data/Metadata.csv" -g "./test_data/AccessoryGene.csv" -s "./test_data/PopulationStructure.csv_labelencoded.csv"

	python PanPred.py -c "create" -i "GYS" -d "1" -m "./test_data/Metadata.csv" -g "./test_data/AccessoryGene.csv" -s "./test_data/PopulationStructure.csv_labelencoded.csv"

	These two above commands will create Feature files for CTX drug using both Gene Information and Year in first case, Gene Information in second one and all information Gene Information, Year & 	Structure in latter case.



2) The above command create input feature file(curated_input_GY.csv) for drug 1(CTX) with features GY of the bacterial strains. Now, 
   we can call predict on the above file and provide our model. The hyperparamaters have already been tuned for the models.

	python3 PanPred.py predict_RF -i "curated_input_GY_1.csv"
	python3 PanPred.py predict_GB -i "curated_input_GY_1.csv" -r 0.2 -n 300
	python3 PanPred.py predict_DL -i "curated_input_GYS_1.csv" -r 0.2 -d 1 -n 400 -m 150 -l 6
	python3 PanPred.py predict_XG -i "curated_input_GYS_1.csv" -r 0.2 -n 500


3) The above command gives a list of scores such as Accuracy, TP, FP, TN etc to measure the model strength. We have used a train_test 
   split ratio of 0.2 which authors have used in their results so as to compare our prediction with theirs. The hyper-paramaters 
   for each model (DL, RF and GB) have been tuned as per the best CV accuracies. 
   We used 5-fold cross validation but the code for this has been removed to maintain only the clean working code.

4) It is observed that RF's and GB's give max accuracies with GY features irrespective of the drug, DL models give max 
   accuracies with G features only and XG Bosst on features from GYS. The Metadata, AccessoryGene and PopulationStructure files remain same always. 





