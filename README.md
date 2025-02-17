# Fashion Classifier

## Installation 💻
You must have Git installed on your machine and either have Python 3.13 or Docker.
### 1. With Docker (Recommended)
Since I have not deployed the Docker image for this project to Docker Hub, you need to create the image yourself.
1. Clone the repo:  
`git clone git@github.com:qhilipp/FashionClassifier.git`
2. cd into the repo:  
`cd FashionClassifier`
3. create the docker image:  
`docker build -t fashion-classifier:latest .`
4. run the image:  
`docker run -v $(pwd):/app fashion-classifier:latest`