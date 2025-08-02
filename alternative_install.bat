@echo off
echo ========================================
echo Car Prediction System - Alternative Installation
echo ========================================

echo.
echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing core packages...
python -m pip install Flask
python -m pip install pymongo
python -m pip install Werkzeug
python -m pip install python-dotenv
python -m pip install dnspython

echo.
echo Installing scientific packages (this may take a while)...
python -m pip install numpy
python -m pip install pandas
python -m pip install joblib

echo.
echo Installing ML packages...
python -m pip install scikit-learn
python -m pip install catboost

echo.
echo Installation completed!
echo.
echo Testing imports...
python -c "import flask; print('Flask: OK')"
python -c "import pymongo; print('PyMongo: OK')"
python -c "import numpy; print('NumPy: OK')"
python -c "import pandas; print('Pandas: OK')"
python -c "import sklearn; print('Scikit-learn: OK')"
python -c "import catboost; print('CatBoost: OK')"

echo.
echo All packages installed successfully!
pause