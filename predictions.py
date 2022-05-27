from main import shipment_pipeline

train_data_path = "train_data.csv"  # provide the train data path
test_data_path = "test_data.csv"  # provide the test data path
models_list = models_list = ["randomforest",
                             "xgboost"]  # models need to be executed
test_size = 0.25

# user can choose own test size and models they want to try. This task contains two models Random Forest Regressor and Xgboost regressor.
# Either of the models can be used or, two models can be used to run the code
results = shipment_pipeline(
    train_data_path, test_data_path, test_size, models_list)
