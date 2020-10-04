[1] CREATE A CONDA ENVIRONMENT:  
    conda create --name speaker python=3.7

[2] INSTALL DEPENDENCIES 
    pip install -r requirements.txt

[3] TRAIN THE MODEL

3.1 train the model using keras feedforward neural network:
    python model.py -o train -m feedforward

3.2 train the model using knn:
    python model.py -o train -m knn

[4] TEST(EVALUATE) THE MODEL

4.1 test the keras feedforward neural network model:
    python model.py -o evaluate -m feedforward -f test_data/speaker2.wav
    python model.py -o evaluate -m feedforward -f test_data/speaker3.wav
    python model.py -o evaluate -m feedforward -f test_data/speaker6.wav
    python model.py -o evaluate -m feedforward -f test_data/speaker8.wav
    python model.py -o evaluate -m feedforward -f test_data/speaker10.wav
    
4.2 test the knn model:
    python model.py -o evaluate -m knn -f test_data/speaker2.wav
    python model.py -o evaluate -m knn -f test_data/speaker3.wav
    python model.py -o evaluate -m knn -f test_data/speaker6.wav
    python model.py -o evaluate -m knn -f test_data/speaker8.wav
    python model.py -o evaluate -m knn -f test_data/speaker10.wav
