
# coding: utf-8
#README



'''run the code using python Assignment_4.py code trains the model and automatically takes in a test image and generates the plots'''
'''IMPORTANT:'''
'''only if you close the plot the next plot will be visible'''
# In[177]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import os


def getPath(s):
    return os.path.join(os.getcwd(),s)
# In[178]:

currentDirectory = os.getcwd()
color = pd.read_csv(os.path.join(currentDirectory,'Dataset/color.csv'))
input_data = pd.read_csv(os.path.join(currentDirectory,"Dataset/input_data.csv"))
data = pd.read_csv(os.path.join(currentDirectory,"Dataset/data.csv"))


# In[179]:


train_gray = input_data[:]
train_color = color[:]

#You can run the below MLPRegressor just once and test on multiple images
# In[180]:
print("training the data.....")

clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(10,15,3), random_state=1, activation='relu', solver='adam')
clf.fit(train_gray, train_color)

#Testing on a new image
#Change the path to the test_image path
# In[181]:

print('testing the model on a image')
img = Image.open(getPath('Expected_Output_Images/1.jpg'))
arr = np.array(img)


# In[182]:


plt.imshow(arr)
plt.savefig('image_original.png')
plt.show()


# In[183]:


#converting the image to gray using the equation given in the assignment writeup
gray = np.array([  np.dot([0.21,0.72,0.07],arr[a][b]) for a in range(arr.shape[0]) for b in range(arr.shape[1]) ])
gray = np.resize(gray,arr.shape[:2])
img = Image.fromarray(gray)
plt.imshow(img)
plt.savefig('image_gray.png')
plt.show()


# In[184]:


#Obtaining the patches from the gray image
data = []
labels = []
#assuming that patch size is always odd
patch_size = 3
k=patch_size//2
for i in range(k,gray.shape[0]-k):
    for j in range(k,gray.shape[1]-k):
        data.append([gray[m][n] for m in range(i-1,i+2) for n in range(j-1,j+2)])
        labels.append(arr[i][j])
        
data = np.array(data)
labels = np.array(labels)


# In[185]:


#making sure that the values are never negative or more than 255
def thresholdLabels(array):
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] < 0:
                array[i][j] = 0
            if array[i][j] > 255:
                array[i][j] = 255
    return array


# In[186]:


final_second_img = clf.predict(data)
final_second_img = thresholdLabels(final_second_img)
final_second_img2 = np.resize(final_second_img,(arr.shape[0]-2*k,arr.shape[1]-2*k,3))
final_second_img2 = final_second_img2.astype(np.uint8)
plt.imshow(final_second_img2)
plt.savefig('image_color.png')
plt.show()

