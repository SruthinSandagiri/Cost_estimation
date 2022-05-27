# Importing the necessary packages
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.preprocessing import MinMaxScaler
import pickle
warnings.filterwarnings("ignore")


class shipment_pipeline():

    def __init__(self, train_data_file_path, test_data_file_path, test_size, models_list):
        # calling the train dataset file into the pipeline
        self.train_data_file_path = train_data_file_path
        # calling the test dataset file into the pipeline
        self.test_data_file_path = test_data_file_path
        self.train_data = self.load_data(self.train_data_file_path)
        #self.train_data = self.preprocess_data(self.train_data_file_path)
        self.test_size = test_size
        self.models_list = models_list
        self.eda_data()
        # self.feature_selection()
        self.data_split()
        self.model_selection()
        self.fit_data()
        self.predict_data()
        self.evaluate_data()
        # self.visualize()
        self.save_model()
        self.eval_test_data()

    def load_data(self, data_file_path):
        # load the train and test data csv files using pandas
        data = pd.read_csv(data_file_path, sep=";",
                           parse_dates=['shipping_date'])
        return self.preprocess_data(data)

    def preprocess_data(self, data):
        # loading the dataframe using pandas and seprator for separating the columns with semincolumns
        #data = pd.read_csv(data_file_path,sep=";")
        # out of the available columns, shipping date column is dropped using below line of code
        data['month'] = data['shipping_date'].dt.month
        data['weekday'] = data['shipping_date'].dt.weekday
        data['year'] = data['shipping_date'].dt.year
        data.drop("shipping_date", axis=1, inplace=True)
        # in the is_adr column of dataframe, it has True as values in every row. to simplify, conversion has been done from bool to int
        data = pd.get_dummies(data, columns=['is_adr'])

        # As the dataset contain the information of latitude and longitude of destination and origin.
        # In order to simplify the learning process, distance is calculated for all the rows using haversine formula
        # after calculating the distance, the distance column is added to dataframe
        data["distance"] = data.apply(lambda row:
                                      haversine((row["origin_latitude"], row["origin_longitude"]),
                                                (row["destination_latitude"], row["destination_longitude"]), unit="km"), axis=1)

        # normalizing the distance values between 0 and 1 (type of min max scaling)
        data["distance"] = data["distance"]/data["distance"].abs().max()

        # rounding the distance values upto 3 decimals
        data['distance'] = data['distance'].apply(lambda x: round(x, 3))
        # after calculating the distance using latitude and longitude information, those columns were dropped
        # data.drop(["origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude"],
        #          axis=1, inplace=True)
        cols_to_norm = ['origin_latitude', 'origin_longitude',
                        'destination_latitude', 'destination_longitude']
        scaler = MinMaxScaler()
        data[cols_to_norm] = scaler.fit_transform(data[cols_to_norm])
        # printing the preprocessed data information
        print(data.info())
        # printing the data descriptive statistics to summarize the data distribution excluding null values
        print(data.describe())
        # returning the preprocessed data for further processing.
        print(data.columns)
        return data

    def eda_data(self):

        # Scatter plot between input features and output to get more insgights about dependent features vs indipendent features
        print("Scatter plot of all independent features against dependent feature")
        # looping through all the available columns, cost has been placed on y axis and remaining columns on x-axis
        for col in self.train_data.columns:
            if col != 'cost':
                plt.figure(figsize=(8, 7))
                plt.xlabel(col)
                plt.ylabel("cost")
                plt.title(col+"vs cost")
                sns.scatterplot(data=self.train_data, x=col,
                                y=self.train_data["cost"])
                plt.plot()

        # Density Distribution plots (to check where the values are concentrated over the interval and their density)
        for col in self.train_data.columns:
            sns.set()
            plt.figure(figsize=(18, 7))
            sns.distplot(self.train_data[col])
            plt.title(col+"Distribution")
            plt.show()

        # Box plot (to Check the whether any outliers present in data using box plot)
        print("Box plot of all features")
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=self.train_data[[
                    "weight", "loading_meters", "is_adr_True", "distance", "cost"]])
        plt.show()
        # Outlier Detection
        print("Percenge of outliers present in every feauture")
        # here in this step, percentage of outliers across all the columns were calculated using "IQR" Method
        for k, v in self.train_data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            inter_q = q3 - q1
            v_col = v[(v <= q1 - 1.5 * inter_q) | (v >= q3 + 1.5 * inter_q)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(self.train_data)[0]
            # printing the percentage of outliers for each column
            print("Column %s outliers = %.2f%%" % (k, perc))

        def remove_outliers(col):
            for x in col:
                upper_limit = self.train_data[x].quantile(0.99)
                lower_limit = self.train_data[x].quantile(0.01)
                new_df = self.train_data[(self.train_data[x] <= upper_limit) & (
                    self.train_data[x] >= lower_limit)]
                self.train_data[x] = np.where(self.train_data[x] >= upper_limit, upper_limit, np.where(
                    self.train_data[x] <= lower_limit, lower_limit, self.train_data[x]))
                return self.train_data
        self.train_data = remove_outliers(["cost", "weight", "distance"])
        # Correlation graph
        # Correlation between variables
        plt.figure(figsize=(15, 10), facecolor='white')
        sns.heatmap(data=self.train_data.corr(), annot=True)

        # visualize pair plot (to understand the best set of features to explain relationship between two variables or
        # pairwise relationship between different variables
        sns.pairplot(data=self.train_data, diag_kind='kde')

    # def feature_selection(self):
        #self.X = self.train_data.iloc[:,self.train_data.columns != 'cost']
        #self.y = self.train_data["cost"]

        #model = ExtraTreesRegressor()
        # model.fit(X,y)
        # print(model.feature_importances_)
        #feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        # feat_importances.nlargest(7).plot(kind='barh')
        # plt.show()

        # return self.X,self.y

    def data_split(self):
        # this function is used to divide the dataset into Dependent variables and indipendent variables
        # Here our Dependent variable is Cost, independent features are rest of columns
        self.X = self.train_data.iloc[:, self.train_data.columns != 'cost']
        self.y = self.train_data["cost"]

        # here independent features and dependent features are further divided into train and test sets with ratio of 70:30, but user
        # have free hand to select split ratio
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y, test_size=self.test_size, random_state=2)
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)

    def model_selection(self):
        # In the modeling part, two models Machine Learning Models used, Random Forest and Xgboost regressor. In future, further models
        # can be added to this pipeline
        self.models = {}
      # This step is used to map our model choice with the parameters, in the form of Key value pairs and added into dictionary
        # when user wants to select only Random Forest Model, model instance is created as randomforest : RandomForestRegressor(parameters) and added to disctionary
        for ml in self.models_list:
            # looping into models list and creating instances of models for training and testing data samples
            if ml == "randomforest":
                self.models[ml] = RandomForestRegressor(max_depth=20, max_features='auto', min_samples_leaf=1,
                                                        min_samples_split=15, n_estimators=700)
            if ml == "xgboost":
                self.models[ml] = XGBRegressor(subsample=0.7999999999999999, n_estimators=500, min_child_weight=1.0, max_depth=15,
                                               learning_rate=0.1, colsample_bytree=0.8999999999999999, colsample_bylevel=0.8999999999999999)

    def fit_data(self):
        # for the created model instances with necessary hyper parameters, the models from the models list were fit with train data
        # In this function, model training is carried out for the training data splits
        for ml in self.models_list:
            # looping through available models and apply model.fit
            self.models[ml].fit(self.X_train, self.y_train)

    def predict_data(self):
        # this function is used predict the Cost parameter for the test data and stored in the dictionary as key value pairs
        self.y_pred = {}
        # looping through available models and apply model.predict
        for ml in self.models_list:
            self.y_pred[ml] = self.models[ml].predict(self.X_test)

    def evaluate_data(self):
        # This function is used to evaluate the model predictions with groud truth test split using multiple metrics
        # mean absolute_error metric
        mae = {}
      # root mean square metric
        rmse = {}
       # r2_score cofficient
        rsquare_score = {}
        # looping into models available from models list
        for ml in self.models_list:
            # finding out metrics for each of both model predictions compared with test data, the storing as key value pair dictionary format
            mae[ml] = mean_absolute_error(self.y_test, self.y_pred[ml])
            rmse[ml] = np.sqrt(mean_squared_error(
                self.y_test, self.y_pred[ml]))
            rsquare_score[ml] = r2_score(self.y_test, self.y_pred[ml])
        # the appended scores are converted as dataframe from dictionary format
        self.results = pd.DataFrame.from_dict([mae, rmse, rsquare_score])
        # creating new row called eval metrics with different metrics names as row values
        self.results["eval_metrics"] = ["mae", "rmse", "rsquare_score"]
        # setting the eval metrics column as index
        self.results.set_index("eval_metrics", inplace=True)
        self.results.plot.bar()
        plt.show()
        print(self.results)
        return self.results

    def visualize(self):
        # this function used to visualize the rmse, mae and r2_scores using graphical format for two models random forest and xgboost
        labels = self.results.columns
        # rounding the scores upto 2 decimals
        mae = round(self.results.loc['mae'], 2)
        rmse = round(self.results.loc['rmse'], 2)
        rsquare_score = round(self.results.loc['rsquare_score'], 2)
        # here bar chart have been used to plot the results using 3 scores as legends
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()  # plotting subplots for each model
        # properties for bar graph for mean_absolute_error
        rects1 = ax.bar(x - width, mae, width, label='MAE')
        # proerties for bar grapg for rmse
        rects2 = ax.bar(x, rmse, width, label='RMSE')
        # properties for bar graph of r2_score
        rects3 = ax.bar(x + width, rsquare_score, width, label='rsquare_score')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Evaluation Metrics')  # y label
        ax.set_title('Evaluation metrics for different regression algorithms')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()

        plt.show()

    def save_model(self):
        # here in this function, the model with highest r2_score is saved as pickle file
        results_list = []
        # appending the both the models r2_score from the dataframe we created for visualizing the different metrics from evaluate data function
        for ml in self.models_list:
            # appending r2_score values in results_list
            results_list.append(self.results.loc["rsquare_score"][ml])
        # looping into results data frame r2_score column values, and selecting the model which has maximum r2 score
        for k, v in self.results.loc["rsquare_score"].items():
            if v == max(results_list):
                # saving the model and its parameters as pickle file which has highest r2_score
                self.file_name = k+".pkl"
                file = open(self.file_name, "wb")
                pickle.dump(self.models[k], file)

    def eval_test_data(self):
        #self.test_data = self.preprocess_data(self.test_data_file_path)
        self.test_data = self.load_data(self.test_data_file_path)
        read_file = open(self.file_name, 'rb')
        best_model = pickle.load(read_file)
        #y_pred = best_model.predict(self.test_data[:100])
        y_pred = pd.DataFrame(data=best_model.predict(
            self.test_data), columns={"cost"})
        y_pred['cost'] = y_pred['cost'].apply(lambda x: round(x, 5))
        y_pred.to_csv("cost_predictions.csv", index=False)
