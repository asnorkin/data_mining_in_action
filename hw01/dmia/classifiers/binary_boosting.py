#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy import optimize


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3, coeff_optimization=False, coeff_reg=1.0):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.estimators_ = []
        
        # Coefficients a_i in prediction sum
        self.coeffs_ = []
        # Regularization coeff to avoid overfitting in a_i selection
        self.coeff_reg = coeff_reg
        # If true optimize every a_i in prediction sum. Else use lr instead
        self.coeff_optimization = coeff_optimization

        
    def loss_grad(self, original_y, pred_y):
        # Вычислите градиент на кажом объекте
        ### YOUR CODE ###  
        return -original_y / (1 + np.exp(pred_y * original_y))# + (1 - original_y) / (1 + np.exp(pred_y))
    

    def fit(self, X, original_y):
        # Храните базовые алгоритмы тут
        self.estimators_ = []        

        original_y = self._refactor_y(original_y)

        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))
            
            # Настройте базовый алгоритм на градиент, это классификация или регрессия?
            ### YOUR CODE ###
            # регрессия
            estimator = deepcopy(self.base_regressor)
            estimator.fit(X, -grad)
            #print estimator

            ### END OF YOUR CODE
            self.estimators_.append(estimator)
            self.coeffs_.append(self._select_coeff(X, original_y, estimator))

        self.out_ = self._outliers(grad)
        self.feature_importances_ = self._calc_feature_imps()

        return self

    
    def _predict(self, X):
        # Получите ответ композиции до применения решающего правила
        ### YOUR CODE ###
        # If zero iteration it returns constant prediction
        if len(self.estimators_) == 0:
            return np.zeros(X.shape[0])
                
        return np.sum(np.array([coeff * est.predict(X) for (est, coeff) in zip(self.estimators_, self.coeffs_)]), axis=0)        
        

    def predict(self, X):
        # Примените к self._predict решающее правило
        ### YOUR CODE ###        
        return (self._predict(X) >= 0).astype(int)

    
    def _outliers(self, grad):
        # Топ-10 объектов с большим отступом
        ### YOUR CODE ###        
        return np.argsort(np.fabs(grad))[-10:]

    
    def _calc_feature_imps(self):
        # Посчитайте self.feature_importances_ с помощью аналогичных полей у базовых алгоритмов
        f_imps = None
        ### YOUR CODE ###
        f_imps = np.sum([est.feature_importances_ for est in self.estimators_], axis=0)

        return f_imps/len(self.estimators_)
    
    
    def _refactor_y(self, y):
        # Algoritm perform {-1, 1} answers
        # If input is {0, 1} marks we change it
        if set(y) == {0, 1}:
            return 2 * (y - 0.5)   
        return y
          
        
    def _select_coeff(self, X, original_y, estimator):
        previous_predictions = self._predict(X)
        last_prediction = estimator.predict(X)
        
        def _loss_coeff(coeff):
            current_predictions = previous_predictions + coeff * last_prediction
            return np.sum(np.log(1 + np.exp(-original_y * current_predictions))) + self.coeff_reg * (coeff ** 2)
        
        result = optimize.minimize_scalar(_loss_coeff)
        if result["success"] and self.coeff_optimization:
            return result["x"]
        
        return self.lr        