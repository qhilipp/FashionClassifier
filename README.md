# Fashion Classifier
A DeepLearning model developed using PyTorch for classifying images from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) data set.
This is my first time using PyTorch and intended to teach me some basic concepts of it.

## Performance 📊
Coming soon ⏰

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
`docker run --name fashion-classifier-container -v $(pwd):/app fashion-classifier:latest python3.13 model.py --epochs 3`

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
`python model.py`

## Arguments 🚀
| Short Argument | Long Argument            | Type   | Description                                                                                             | Default Value |
|----------------|--------------------------|--------|---------------------------------------------------------------------------------------------------------|---------------|
| `-d`           | `--device`               | str    | The device on which PyTorch should perform all Tensor calculations. Defaults to `cpu` if not available. | `'cpu'`       |
| `-e`           | `--epochs`               | int    | The number of epochs used to train the model.                                                           | `20`          |
| `-l`           | `--learnrate`            | float  | The learning rate for the optimizer.                                                                    | `0.001`       |
| `-b`           | `--batchsize`            | int    | The batch size for the data loader.                                                                     | `64`          |
