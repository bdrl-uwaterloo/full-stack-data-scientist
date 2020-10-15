[1] CREATE A CONDA ENVIRONMENT: 
```sh
conda create --name speaker python=3.7
```
[2] INSTALL DEPENDENCIES 
```sh
 pip install -r requirements.txt
```
[3] TRAIN THE MODEL

3.1 train the model using random forest:
```sh
python model.py -o train -m random_forest
```
3.2 train the model using linear regression:
```sh
python model.py -o train -m linear
```
[4] User's Interfacec

4.1 RUN THE SERVER
```sh
python server.py
```
4.2 Follow the instruction: choose and upload test.csv


