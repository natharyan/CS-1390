# implement regression model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
from data_preprocessing import preprocess_data_regression

class LinearRegression:
    """
    This is my implementation of multiple linear regression for predicting
    fuel consumption based on engine size and coemsissions.
    I have used the following methods in my implementation:
    - pred: returns the prediction for a single datapoint
    - mse: returns the mse over the dataset
    - calc_grad: calculates the gradient for gradient descent by doing partial integrations over each weight coefficient
    - fit: fit the model to the training dataset
    - loss_plot: plot the mse loss plot over the training and validation data
    """
    def __init__(self, alpha=0.0001, num_iterations=30):
        self.alpha = alpha
        self.num_iterations = num_iterations
        # theta is a vector of theta_0, theta_1 ... theta_n for n number of features
        self.theta = []
        self.mse_history = []
        self.rmse_history = []
        self.rsquared_history = []

    def pred(self, X):
        # X is a single tuple, not an array of tuples
        return self.theta[0] + sum([(self.theta[i+1])*X.iloc[i] for i in range(len(X))])

    def mse(self, X_data, Y_true):
        mse = 0
        for i in range(len(X_data)):
            mse += (self.pred(X_data.iloc[i]) - Y_true.iloc[i])**2
        # return mean of the squared errors
        return mse / len(X_data)
    
    def rmse(self, X_data, Y_true):
        return np.sqrt(self.mse(X_data, Y_true))
    
    def r_squared(self, X_data, Y_true):
        y_mean = sum(Y_true)/len(Y_true)
        ss_res = sum([(Y_true.iloc[i] - self.pred(X_data.iloc[i]))**2 for i in range(len(X_data))])
        ss_tot = sum([(Y_true.iloc[i] - y_mean)**2 for i in range(len(Y_true))])
        return 1 - ss_res/ss_tot

    def calc_grad(self,X,Y):
        # loss = sigma((y_n - (theta_0 + theta_1*x_n))^2)
        # taking partial derivatives of the loss function above with respect to theta_0 and theta_1
        grad_theta0 = sum([-2*(Y.iloc[i] - (self.theta[0] + sum([self.theta[k+1]*X.iloc[i][k] for k in range(X.shape[1])]))) for i in range(len(X))])
        self.theta[0] = self.theta[0] - self.alpha*grad_theta0
        for i in range(X.shape[1]):
            # gradcur_theta = sum([-2*(Y.iloc[i] - (self.theta_0 + self.theta_1*X[i])) for i in range(len(X))])
            # calculate the gradient of the loss function with respect to theta_{i+1}
            gradcur_theta = sum([-2*X.iloc[j][i]*(Y.iloc[j] - (self.theta[0] + sum([self.theta[k+1]*X.iloc[j][k] for k in range(X.shape[1])]))) for j in range(len(X))])
            self.theta[i+1] = self.theta[i+1] - self.alpha*gradcur_theta

    def fit(self, X_train, y_train,X_validation,y_validation):
        # create the theta vector with random initialization, of size dimensions(X)
        self.theta.append(0.0)
        for i in range(X.shape[1]):
            # self.theta.append(np.random.rand(1)[0])
            self.theta.append(0.0)

        for i in range(self.num_iterations):
            train_loss = self.mse(X_train, y_train)
            validation_loss = self.mse(X_validation,y_validation)
            self.mse_history.append((train_loss, validation_loss))
            self.calc_grad(X_train, y_train)
            train_loss = self.rmse(X_train, y_train)
            validation_loss = self.rmse(X_validation,y_validation)
            self.rmse_history.append((train_loss, validation_loss))
            train_loss = self.r_squared(X_train, y_train)
            validation_loss = self.r_squared(X_validation,y_validation)
            self.rsquared_history.append((train_loss, validation_loss))

    def loss_plot(self):
        train_loss = [self.mse_history[i][0] for i in range(len(self.mse_history))]
        validation_loss = [self.mse_history[i][1] for i in range(len(self.mse_history))]
        plt.plot(train_loss,label='train')
        plt.plot(validation_loss, label='validation', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('MSE')
        plt.legend()
        plt.show()
        train_loss = [self.rmse_history[i][0] for i in range(len(self.rmse_history))]
        validation_loss = [self.rmse_history[i][1] for i in range(len(self.rmse_history))]
        plt.plot(train_loss,label='train')
        plt.plot(validation_loss, label='validation', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('RMSE')
        plt.legend()
        plt.show()
        train_loss = [self.rsquared_history[i][0] for i in range(len(self.rsquared_history))]
        validation_loss = [self.rsquared_history[i][1] for i in range(len(self.rsquared_history))]
        plt.plot(train_loss,label='train')
        plt.plot(validation_loss, label='validation', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('R^2')
        plt.legend()
        plt.show()

    def plotline(self, X, Y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X['COEMISSIONS '], X['ENGINE SIZE'], Y)
        x1 = np.linspace(X['COEMISSIONS '].min(), X['COEMISSIONS '].max(), 100)
        x2 = np.linspace(X['ENGINE SIZE'].min(), X['ENGINE SIZE'].max(), 100)
        x1, x2 = np.meshgrid(x1, x2)
        z = self.theta[0] + self.theta[1] * x2 + self.theta[2] * x1
        ax.plot_surface(x1, x2, z, color='red', alpha=0.5)
        ax.set_xlabel('ENGINE SIZE')
        ax.set_ylabel('COEMISSIONS')
        ax.set_zlabel('FUEL CONSUMPTION')
        plt.title(f'Regression Plane: theta_0={self.theta[0]:.3f}, theta_1={self.theta[1]:.3f}, theta_2={self.theta[2]:.3f}')
        plt.show()

    def predict(self,X):
        # TODO: ask if X is going to be a dataframe/list or a single feature vector
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.pred(X.iloc[i]))
        return predictions

if __name__ == "__main__":
    # the following code extracts the arguments passed in the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path for saved model')
    parser.add_argument('--data_path', type=str, required=True, help='path for evaluation data')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='path to save evaluation metrics')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='path to save regression predictions')

    arguments = parser.parse_args()

    # load the required trained model
    with open(arguments.model_path, 'rb') as file:
        regression_model = pickle.load(file)
    
    # get the evaluation data from the passed path
    evaluation_data = pd.read_csv(arguments.data_path)

    # do data preprocessing
    df, X, y = preprocess_data_regression(arguments.data_path)
    predictions = regression_model.predict(X)

    mse = regression_model.mse(X, y)
    rsme = regression_model.rmse(X, y)
    r_squared = regression_model.r_squared(X, y)


    # save the evaluation metrics
    with open(arguments.metrics_output_path, 'w') as file:
        file.write('Regression Metrics:\n')
        file.write(f'Mean Squared Error (MSE): {mse}\nRoot Mean Squared Error (RMSE): {rsme}\nR-squared (R2) Score: {r_squared}')

    # save the model predictions
    with open(arguments.predictions_output_path, 'w') as file:
        file.write('Dataframe Index,Prediction\n')
        for i in range(len(predictions)):
            file.write(f'{i},{predictions[i]}\n')
    print("evaluation metrics and predictions saved successfully...")