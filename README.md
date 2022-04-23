18/19 - April - 2022
----
## Artifical intillegence
To make computers think like humans.
Machine learning (ML) and deep learning (DL) are also approaches to solving problems. The difference between these techniques and a Python script is that ML and DL use training data instead of hard-coded rules, but all of them can be used to solve problems using AI

## Deep learning
Deep learning is a technique in which you let the neural network figure out by itself which features are important instead of applying feature engineering techniques. This means that, with deep learning, you can bypass the feature engineering process.

## Feature engineering
Used to train a model to predict the sentiment in a sentence? Or what if you have an image, and you want to know whether it depicts a cat?

Another name for input data is feature, and feature engineering is the process of extracting features from raw data.
- Not having to deal with feature engineering is good because the process gets harder as the datasets become more complex.

## Neural networks
It is a system that learns to predict output by:
1. taking input data
2. make a prediction
3. compare it to expected output
4. adjust internal state to predict correctly.

Vectors, layers, and linear regression are some of the building blocks of neural networks. The data is stored as vectors, and with Python you store these vectors in arrays.

Each layer transforms the data that came from the previous layer by applying some mathematical operations.

```
python -m pip install ipython numpy matplotlib

ipython
```

[AI, Neural network, Deep learning, Feature engineering](https://realpython.com/python-ai-neural-network/)
_____

## TensorFlow
Tenserflow is a library used to build neural networks, its used for machine learning applications easily with an API called Keras.
tensorflow was mainly used for detecting ard recognising (like traffic lights/pedestrian).
 - Applications of Tensorflow:
1. Analyzing rush-hour (peak) traffic
2. Predicting car accident risk
3. Greater knowledge of on-road conditions - like weather change
4. Driving behavior over specific roads-  for example, it uses Flow/Density models to calculate relative speeds
5. The effectiveness of safety measures

[Road safety-Tensorflow](https://www.datasciencecentral.com/how-tensorflow-is-helping-in-maintaining-road-safety/)

[Tensorflow tutorial](https://www.youtube.com/watch?v=6_2hzRopPbQ&ab_channel=NicholasRenotte)

- Keras is used as a high-level API to build and train models in TensorFlow.  [Modules](https://www.tensorflow.org/api_docs/python/tf/keras)

[Quick start Keras](https://www.tensorflow.org/tutorials/keras/classification)

[Quick start](https://www.tensorflow.org/tutorials/quickstart/beginner)


- [0:00:00](https://www.youtube.com/watch?v=6g4O5UOH304&t=0s) What is a Neural Network?
- [0:26:34](https://www.youtube.com/watch?v=6g4O5UOH304&t=1594s) How to load & look at data
- [0:39:38](https://www.youtube.com/watch?v=6g4O5UOH304&t=2378s) How to create a model
- [0:56:48](https://www.youtube.com/watch?v=6g4O5UOH304&t=3408s) How to use the model to make predictions
- [1:07:11](https://www.youtube.com/watch?v=6g4O5UOH304&t=4031s) Text Classification (part 1)
- [1:28:37](https://www.youtube.com/watch?v=6g4O5UOH304&t=5317s) What is an Embedding Layer? Text Classification (part 2)
- [1:42:30](https://www.youtube.com/watch?v=6g4O5UOH304&t=6150s) How to train the model - Text Classification (part 3)
- [1:52:35](https://www.youtube.com/watch?v=6g4O5UOH304&t=6755s) How to saving & loading models - Text Classification (part 4)
- [2:07:09](https://www.youtube.com/watch?v=6g4O5UOH304&t=7629s) How to install TensorFlow GPU on Linux



_____

## API
API stands for application programming interface, which is a set of definitions and protocols for building and integrating application software.

This interface allows different systems to talk to each other without having to understand exactly what each other does.

- SOAP vs REST vs GraphQL
1. SOAP (Simple Object Access Protocol) is typically associated with the enterprise world, has a stricter contract-based usage, and is mostly designed around actions.
2. REST (Representational State Transfer) is typically used for public APIs and is ideal for fetching data from the web. It’s much lighter and closer to the HTTP specification than SOAP.
3. GraphQL. Created by Facebook, GraphQL is a very flexible query language for APIs, where the clients decide exactly what they want to fetch from the server instead of the server deciding what to send.

When consuming APIs with Python, there’s only one library you need: requests.

> import requests

> requests.get("https://randomuser.me/api/")

[API's](https://realpython.com/python-api/)
____
## Pygame
python library used for making multimedia applications like games, art, music, sound, video and multimedia projects. 

[Official webpage](https://www.pygame.org/news)

[Tutorial](https://realpython.com/pygame-a-primer/)
____
## Simpy
Python framework for event simulation. 

SimPy is a process-based discrete-event simulation framework based on standard Python.

Processes in SimPy are defined by Python generator functions and may, for example, be used to model active components like customers, vehicles or agents. SimPy also provides various types of shared resources to model limited capacity congestion points (like servers, checkout counters and tunnels).
[SimPy](https://simpy.readthedocs.io/en/latest/)

[Tutorial](https://realpython.com/simpy-simulating-with-python/)

[Reference](https://www.youtube.com/watch?v=6g4O5UOH304&ab_channel=freeCodeCamp.org)
_____
## Time - timeit()
in general timeit is a method in python library used to measure the execution time taken by the given code snippet. The Python library runs the code statement 1 million times and provides the minimum time taken from the given set of code snippets.
 
> Project

timeit module can be used to calculate the delay between each two cars.
in our project, we will take a fixed time from specific data and compare it with the result of timeit library. 

[Timeit()](https://docs.python.org/3/library/timeit.html)

## Time - sleep()
Python time sleep function is used to add delay in the execution of a program. We can use python sleep function to halt the execution of the program for given time in seconds. Notice that python time sleep function actually stops the execution of current thread only, not the whole program.

```
import time

print("Before the sleep statement")
time.sleep(5)
print("After the sleep statement")
```
If you run the above code then you will see that the second print executes after 5 seconds. So you can make delay in your code as necessary.
[Sleep](https://www.journaldev.com/15797/python-time-sleep#:~:text=Python%20time%20sleep%20function%20is,only%2C%20not%20the%20whole%20program.)
____
## Sumolib 
sumolib is a set of python modules for working with sumo networks, simulation output and other simulation artifacts.

importing sumolib:
To use the library, the <SUMO_HOME>/tools directory must be on the python load path. This is typically done with a stanza like this:
```
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
```
> Project
we can use this library to design a simulation of the traffic AI System
_____
## AI Traffic ligths control system 
The basic idea around this topic is to control the opening and closing time of traffic lights according to number of vehicles passing through each one.
>Project
we are going to deal with a T-section street road with 5 traffic lights, input mainly will be vehicles volume (number of cars) and the output will be the score of numebr of cars passing through a specific period of time**.

[Example](https://www.youtube.com/watch?v=CQu4wFLC79U&t=177s&ab_channel=MichaelRechtin)
[Traffic detection using tensorflow](https://builders.intel.com/docs/aibuilders/traffic-light-detection-using-the-tensorflow-object-detection-api.pdf)

_____
## Linear regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.

[Linear regression](https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/)

