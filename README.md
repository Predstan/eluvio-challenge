# Challenge
  The dataset is tabular and the features involved should be self-explanatory. We would like for you to come up with a specific problem yourself and solve it properly. This is an “open challenge,” mainly focusing on natural language processing. The problem could be either about predictive modeling or providing analytical insights for some business use cases. Note the problem should be treated as large-scale, as the dataset is large (e.g., >100GB) and will not fit into the RAM of your machine. Python is strongly recommended in terms of the coding language. 

### Problem Formation
#### Suggest Title of news for Reader Based on Interest and Interested Topic


### Methodology
To solve this Problem, I will be building an embedding Model from the Dataset to disparately show the similarities between the titles available on the Dataset which will then be Fed into a Kmeans Algorithm(Unsupervised Learning) to create Clusters which we can use to model a recommendation system. However, to kickstart this project and due to time limits, I will use available models like the Tensorflow's Sentence Encoder and the Gensim Word2vec Model as my embeddings

The Problem formation is a bit complicated since there are no explicit label to use on the data, so i am creating a label with a kmeans cluster algorithm.
The steps taken to achieve this is as follows:
1. Create a Sentence Embeddings: I will be using the Tensorflow's sentence Encoder due to time constraints and will not be buiding a sentece Encoder for a start. Tensorflow's sentence encoder is available at https://tfhub.dev/google/universal-sentence-encoder/4. The output is a 512 long 1 dimension vector for each sentence fed into the network


2. Label data with Kmeans Algorithm: I find the optimal number of cluster for the dataset. However, i have considered limiting the number of Cluster to a maximum of 100. This is due to computational power that is required as number of cluster increases. We find the OPtimal number between 10 - 100

3. Build a prediction Model: This is very optional, However, to reduce the inference time since we have to find the distance to all the clusters to find where a sentence belong, we are doing this to help us reduce that time. A model will learn and capture the weigths that predicts the cluster where a sentence should belong

4. After Prediction, we only compare the sentence embeddings in that cluster and recommend the top 5 news ranked by earlier dates and Up_votes. This is very important since we assume readers will only be interested in newest news with highest up_votes

## DEMO
### Requirements:
1. Python 3.6+
2. Pandas
3. Numpy
4. Tensorflow 2.4+ with tf.compat.v1 available for training

To Run predictions with the Elivio DataSet or to train a new model. Please Read through this Section. 
## Step 1
### CLONE REPOSITORY
Open your Command Line or Terminal depending on your Operating System. You may create a virtual environment to isolate this project files from your local machine;
Read here for virtual environment :https://docs.python.org/3/library/venv.html
But for simplicity, clone this repository by runnung the command below in the terminal line by line
```
$ mkdir recommendation
$ cd recommendation
$ git clone git@github.com:Predstan/eluvio-challenge.git
$ cd eluvio-challenge
```

## Step 2:
### Unzip the Model Directory 
```
$ unzip Model.zip
$ rm Model.zip
```
## Step 3: 
### Run prediction
```
$ ./recommend.py <Enter Your Sentence Here>
```
You should enter your sentence to recommend title in line with the command above. For a start you can try words below:
1. Syria
2. Putin
3. Russia
4. Samsung
or any sentence of your choice and Model will predict the top 50 by earliest date and Highest Vote

# Training a new Model
### Follow implementation in the Jupyter Notebook note.ipynb
