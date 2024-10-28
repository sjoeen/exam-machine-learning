import numpy as np
import math





class LogisticRegression():
    """
    This class does logistic regression. The weight and biases starts as zero
    and gets adjusted accoringly. 
    """
    
    def __init__(self,lr,epochs,treshold):

        self.epochs = epochs
        self.lr = lr
        self.weights = 0
        self.bias = 0
        self.treshold = treshold
        

    def fit(self,X,Y):
        """
        This function adjust and fits the weight and biases to fit the data,
        using gradient decent, in order to use gradient decent we use the derivative
        of the loss function in respect to W and B. 

        input: data
        return: updates weight(s) and bias
        """
        n,features = X.shape
        self.weights =  np.zeros(features,dtype=np.float64)
            #makes the weights a matrix with the 
            #same lenght as amount of features.

        for _ in range(self.epochs):

            logistic_hat = self.predict(X)
            error = logistic_hat - Y
            der_weight = -(2/n) * np.dot(X.T,error)
            der_bias = -(2/n) * np.sum(error)
                #derviatives in respect to weight and bias, 
                # full explaination in report.
            
            der_weight = np.asarray(der_weight, dtype=np.float64)
                #had some type issues, this line fixed it. 

            self.weights +=  self.lr * der_weight
            self.bias += self.lr * der_bias
                #updates the weight(s) and bias usings the 
                #opposite direction of the derivative


        return self.weights,self.bias

 
    def predict(self,X):
        """
        This function makes the predictions both linear and logistic
        """

        linear_hat = np.dot(X,self.weights) + self.bias
            #Since X is a matrix, it's neccesary to do matrix multiplication. 
        logistic_hat = self.sigmoid(linear_hat)
            #pass the W * x + b into the x variable of the function.
        
        return (logistic_hat >= self.treshold).astype(int)
    

    def sigmoid(self,x): 
        """
        This functions caps the range [0,1].
        """

        x = np.asarray(x, dtype=np.float64)
            #got typeerror, this line fixed it

        x = np.clip(x, -500, 500)
            #had some value issues, this seemed to fix it. 
        return 1 / (1 + np.exp(-x))


    def evalute(self,X_test,Y_test):
        """
        This function uses the current weight(s) and bias and 
        makes prediction using these values. It then compares
        the machine predictions to the labels of the test set.

        input: test data set
        return: accuracy, confusion matrix
        """

        n,features = X_test.shape
        predictions = []
                        
        logistic_hat = self.predict(X_test)

        for numbers in logistic_hat:
            if numbers <= self.treshold:
                predictions.append(0)
            else:
                predictions.append(1)

        predictions = np.array(predictions)
            #convert into array

        equal_elements = predictions == Y_test
        successful_guesses = np.sum(equal_elements)
            #Count how many times they have equal numbers at equal indices
            #chatgpt helped me with this. 

        confusion_matrix = (self.confusion_matrix(predictions,Y_test))


        accuracy = successful_guesses/n

        return round(accuracy,3),confusion_matrix
    

    def confusion_matrix(self,predictions,Y_test):
        """
        returns the amount of true positive,false positive,true negative and false negative.
        This function is used withing the evaluate function, therefore this is just a 
        helping function.
        This is made as a seperate function to make the code more readable.

        input: predictions, Y_test

        returns: 
             array of [[TN,FP],[FN,TP]] 
        """

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        for index in range(len(predictions)):
            if predictions[index] == 1:
                if Y_test[index] == 1:
                    TP += 1
                else:
                    FP += 1
            if predictions[index] == 0:
                if Y_test[index] == 0:
                    TN += 1
                else:
                    FN += 1

        confusion_matrix = np.array([[TN, FP],
                                     [FN, TP]])
        return confusion_matrix


    def set_params(self, **params):
        """
        chatgpt made this function in order to make my implementation compatible with
        the cross validation method. 
        https://chatgpt.com/share/671a843c-b380-800d-8d4a-e1a040492986
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def get_params(self, deep=True):
        return {"lr": self.lr, "epochs": self.epochs, "treshold": self.treshold}

    


