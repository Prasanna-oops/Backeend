from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

class PredictView(APIView):
    def post(self, request):
    # Extract input values from request
        v1 = request.data.get('v1')
        v2 = request.data.get('v2')
        v3 = request.data.get('v3')
        v4 = request.data.get('v4')
        # Check if any value is None
        if None in [v1, v2, v3, v4]:
            return Response({'error': 'Some values are missing'}, status=status.HTTP_400_BAD_REQUEST)
        # Convert values to float
        try:
            v1 = float(v1)
            v2 = float(v2)
            v3 = float(v3)
            v4 = float(v4)
        except ValueError:
            return Response({'error': 'Some of the values are not valid numbers'}, status=status.HTTP_400_BAD_REQUEST)
        # Load the trained model and preprocess data
        data = pd.read_csv('C:\Front-Back\Streamlit\Prasanna raj KJ_717822I239 (1).csv') # Update with your file path
        data=data.replace(np.nan,0)
        x = data.iloc[:, [0, 2, 3, 5]]
        y = data.iloc[:, [1]]
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=1)
        # Train the model
        a = LinearRegression()
        a.fit(xtrain, ytrain)
        p=PolynomialFeatures(degree=6)
        x_poly=p.fit_transform(x)
        a.fit(x_poly,y)
        # Make predictions
        out = a.predict(np.array([v1, v2, v3, v4]).reshape(1, -1))
        prediction = int(out[0])
        # Return the prediction
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)