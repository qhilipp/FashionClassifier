# Fashion Classifier
A DeepLearning model developed using PyTorch for classifying images from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) data set.
This is my first time using PyTorch and intended to teach me some basic concepts of it.

## Performance 📊
I tuned the hyper-parameters for 20 trials with 5 epochs each and then trained the model for 25 epochs.
This resulted in an accuracy of 93.19% for unseen test data.
![Trials](graphs/Trials.png) ![Loss](graphs/Loss.png)
The following graph shows a sample of the test data along with the models predictions.
![Sample](graphs/Sample.png)
As you can see, the model is pretty confident for all predictions that it got correct, but also relatively confident for the one 
prediction that it got wrong, which I assume is due to the similarity of Shirts and Pullovers.

## Installation 💻
### Option 1 with Docker 🐳 (Recommended)
Since I have not deployed the Docker image for this project to Docker Hub, you need to create the image yourself.
1. Clone the repo:  
`git clone git@github.com:qhilipp/FashionClassifier.git`
2. cd into the repo:  
`cd FashionClassifier`
3. Create the docker image:  
`docker build -t fashion-classifier:latest .`
4. Run the image:  
`docker run --name fashion-classifier-container -v $(pwd):/app fashion-classifier:latest`  
or you can specify arguments for the model like so:  
`docker run --name fashion-classifier-container -v $(pwd):/app fashion-classifier:latest python3.13 main.py --epochs 3`

### Option 2 without Docker 🐳❌
Make sure you have Python 3.13 installed and the 'python' command pointing to version 3.13.
1. Clone the repo:  
`git clone git@github.com:qhilipp/FashionClassifier.git`
2. cd into the repo:  
`cd FashionClassifier`
3. Create a virtual environment  
`python -m venv env`
4. Activate it  
`source evn/bin/activate`
5. Install the packages  
`pip install -r requirements.txt`
6. Run  
`python main.py`

## Arguments 🚀
| Argument                    | Kurzform | Typ   | Beschreibung                                                                                                    | Standardwert |
|-----------------------------|----------|-------|-----------------------------------------------------------------------------------------------------------------|--------------|
| `--load`                    | `-l`     | str   | Der Name der Datei, aus der das Modell geladen werden soll.                                                     | `None`       |
| `--save`                    | `-s`     | str   | Der Name der Datei, in der das Modell gespeichert werden soll.                                                  | `None`       |
| `--device`                  | `-d`     | str   | Das Gerät, auf dem PyTorch die Tensorberechnungen ausführen soll (`cpu` oder `cuda`, falls verfügbar).          | `'cpu'`      |
| `--epochs`                  | `-e`     | int   | Die Anzahl der Epochen, die zum Trainieren des Modells verwendet werden.                                        | `20`         |
| `--trials`                  | `-t`     | int   | Die Anzahl der Versuche, um optimale Hyperparameter zu finden.                                                  | `5`          |
| `--trial_epochs`            | `-te`    | int   | Die Anzahl der Epochen pro Versuch beim Finden optimaler Hyperparameter.                                        | `2`          |
| `--load_hyper_parameters`   | `-lh`    | str   | Der Name einer JSON-Datei, aus der das Modell seine Hyperparameter laden soll.                                  | `None`       |
| `--save_hyper_parameters`   | `-sh`    | str   | Der Name einer JSON-Datei, in der das Modell seine Hyperparameter speichern soll.                               | `None`       |

