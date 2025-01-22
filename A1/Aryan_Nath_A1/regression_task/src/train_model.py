# implement regression model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
import time
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
        return self.theta[0] + sum([self.theta[i+1]*X[i] for i in range(len(X))])

    def mse(self, X_data, Y_true):
        mse = 0
        for i in range(len(X_data)):
            # get the squared error for each data point w.r.t. the mean
            mse += (self.pred(X_data.iloc[i]) - Y_true.iloc[i])**2
        # return mean of the squared errors
        return mse / len(X_data)
    
    def rmse(self, X_data, Y_true):
        # return the square root of the mse
        return np.sqrt(self.mse(X_data, Y_true))
    
    def r_squared(self, X_data, Y_true):
        # calculate the r squared value
        # mean of the true values
        y_mean = sum(Y_true)/len(Y_true)
        # sum of squareds of difference residuals w.r.t. the true values
        ss_res = sum([(Y_true.iloc[i] - self.pred(X_data.iloc[i]))**2 for i in range(len(X_data))])
        # total sum of squares of the difference of true values w.r.t. the mean
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
            # get the loss values for each iteration, and append them to their respective arrays
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
        predictions = [self.pred(x) for x in X]
        return predictions

# df = pd.read_csv('data/fuel_train.csv')

# # filtered the datafram to remove rows with missing engine size or fuel consumption values
# df = df.sample(frac=1).reset_index(drop=True)

# df = df.dropna(subset=['ENGINE SIZE', 'FUEL CONSUMPTION'])

# # using engine size and coemissions
# X = df[['ENGINE SIZE','COEMISSIONS ']]
# y = df['FUEL CONSUMPTION']

# # standardize the data
# for i in range(X.shape[1]):
#     mean = sum(X.iloc[:,i])/len(X.iloc[:,i])
#     X.iloc[:,i] = (X.iloc[:,i] - mean)/np.std(X.iloc[:,i])
#     # X.iloc[:,i] = X.iloc[:,i]/max(X.iloc[:,i])

df, X, y = preprocess_data_regression('data/fuel_train.csv')

X_train = X[:int(0.8*len(X))]
y_train = y[:int(0.8*len(y))]


X_validation = X[int(0.8*len(X)):]
y_validation = y[int(0.8*len(y)):]

print(len(X_train))
print(len(X_validation))


# linear_regressor = LinearRegression(alpha=0.000001, num_iterations=4000) # 0.5
# linear_regressor = LinearRegression(alpha=0.000001, num_iterations=6000) # 0.4
linear_regressor = LinearRegression(alpha=0.000001, num_iterations=6200) # 0.24 on the validation data
# linear_regressor = LinearRegression(alpha=0.000001, num_iterations=7500) # overfit

print("Training started...")
start = time.time()
linear_regressor.fit(X_train, y_train,X_validation,y_validation)
print("Training completed")

print("Training time:", )
print(time.time()-start)
print("Plotting loss...")
linear_regressor.loss_plot()

print("Training MSE (Training,Validation):",linear_regressor.mse_history[-1])
print("Training RMSE (Training,Validation):",linear_regressor.rmse_history[-1])
print("Training R squared (Training,Validation):",linear_regressor.rsquared_history[-1])

print("Plotting regression plane...")
linear_regressor.plotline(X_train, y_train)
linear_regressor.plotline(X_validation, y_validation)

# print("Saving model...")
# with open('models/regression_model_final.pkl', 'wb') as f:
#     pickle.dump(linear_regressor, f)