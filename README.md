This was a group assignment done by 
CECIL OCHIENG: IN14/00022/18
GEORGE HINGA: IN14/00023/20
EMMANUEL KERAGE: IN14/00063/20

METHODOLOGY
i.	Introduction
For the project we were required to develop a neural network model that accurately classifies different images into their specified categories. This will involve the designing of a convolutional neural network (CNN) that has an output of 10 classes.
ii.	Dataset description
Fashion-MNIST is a dataset comprising of 28×28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images and the test set has 10,000 images. Fashion-MNIST shares the same image size, data format and the structure of training and testing splits with the original MNIST. The datasets are loaded individually in the model to begin the learning process.
 
iii.	Data preprocessing
Dataset splitting: The dataset used for machine learning was partitioned into two
subsets — training and test sets. We split the dataset into two with a split ratio of 70%
i.e., in 100 records 70 records were a part of the training set and remaining 30 records were a part of the test set. The hyper parameters for training included a sample size of 256 before updating the weights, 10 output classes and 50 iterations during the training.
 
Data reshaping: The images in the data set were reshaped to 28 by 28 pixels to ensure they are compatible to for in the model by using the numpy module.
 
iv.	Model development
To develop the model, we started by initializing it using the Keras deep learning library. The model is made up an input convolutionary layer and two pooling layers and an output layer. Pooling layers down sample the features obtained from the convolutionary layers reducing the spatial dimensions and computing in the network. A ReLU activation function is used with normal initialization weights including dropout layers to prevent overfitting. When it comes to the output layers, flattening the output of the last pooling layers to the ten classes that are required in the labeling of the individual clothing. In the output later a softmax activation function is used as the model is a multiclass classifier. Model compilation is done using the Adam optimizer and training can now commence.
 
Summary of the model

v.	Model training 
After processing the collected data and split it into train and test can we proceeded with a model training. This process entails feeding the algorithm with training data. An algorithm will process data and output a model that is able to find a target value (attribute) in new data an answer you want to get a predictive analysis. The purpose of model training is to develop a model. 70% of the Fashion MNIST dataset was used for training. The number of epochs used for the training were 50 and batch size of 256 samples and monitoring the learning rate and dropout rate of the model.
 
vi.	Results obtained
After running all the iterations and training is completed, model evaluation begins where metrics such as accuracy that shows how well the model performs. This enables fine tuning of the model’s hyper parameters and trying different architectures to improve the performance. The models performance was monitored through continuous training with different sizes of training and testing data to obtain the optimum model performance. 
 
The overall accuracy of the model was 89% which shows that the model has been well trained to make predictions and acquire the desired results. To demonstrate the models accuracy a test dataset was used to make predictions for analysis of the predicted outcomes and desired output. The test data was 30% of the total Fashion MNIST dataset.
 
The grid above contains predictions made from the test data.
vii.	Challenges and recommendations
The main challenge faced was during exploration and exploitation where we tried different data size inputs to obtain the most accurate combination of training and testing data. This was time consuming as it required numerous trials to get the desired results. The large dataset took a significant amount of time to be processed during training and testing as it was required to improve the accuracy of the model.
To improve upon the models performance, we can utilize the adaptive learning rate algorithms like Adam to automatically adjust learning rates based on the gradient magnitude. Another way of improving the performance is by leveraging transfer learning techniques to transfer knowledge from related tasks to pre-trained models to accelerate learning in new environments. Using different exploration strategies such as Upper Confidence Bound to encourage exploration while gradually shifting towards exploitation as the model learns.

