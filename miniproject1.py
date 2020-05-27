from collections import Counter
import numpy as np
import timeit
import matplotlib.pyplot as plt
#%matplotlib inline
import json # we need to use the JSON package to load the data, since the data is stored in JSON format


with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 


# Preprocessing
def preprocess(txt):
  out = txt.lower().split()
  return out
  
for datapt in data:
  for k, v in datapt.items():
    if k == "text" and isinstance(v, str): 
      datapt[k] = preprocess(v)  

# Matrix X
def matrixX(dataset, top_words):
  size = len(dataset)
  allwords = []
  
  for datapt in dataset:
    allwords.extend(datapt["text"])  
  
  counter = Counter(allwords)

  popwords = counter.most_common(top_words)

  wordlist = []
  for w,num in popwords:
    wordlist.append(w)
    
  for datapt in dataset:
    datapt["xcounts"]=np.zeros(top_words)
    for wrd in datapt["text"]:
      if wrd in wordlist:
        index = wordlist.index(wrd)
        datapt["xcounts"][index] = (datapt["xcounts"][index]+1)
         
  X = np.zeros((size,top_words+4))
  i = 0

  for datapt in dataset:
    if datapt["is_root"] == False:
      datapt["is_root"]=0
    else: 
      datapt["is_root"]=1
    
    X[i, 0]=1
    X[i, 1]=datapt["controversiality"]
    X[i, 2]=datapt["children"]
    X[i, 3]=datapt["is_root"]
    p=4
    for val in datapt["xcounts"]:
      X[i, p]=val
      p=p+1
   
    i=i+1  
  return X
    
# Vector Creation y

def vectorY (dataset):
  size = len(dataset)
  y = np.zeros((size,1))
  i2 = 0

  for datapt in dataset:
    y[i2,0] = datapt["popularity_score"]
    i2=i2+1

  return y

# implementing model with extra features

#Task 2: Implement linear regression

#Closed form solution


def closed_form_solution(dataset, top_words):
  #start = timeit.default_timer()
  size = len(dataset)
  X = matrixX(dataset,top_words)
  y = vectorY(dataset)

  XTX = np.dot(X.T, X)
  XTy = np.dot(X.T, y)
  w = np.dot(np.linalg.inv(XTX), XTy)
  error = np.dot(np.transpose(y-np.dot(X,w)),y-np.dot(X,w))
  MSE = error/size
  print("MSE CLOSED FORM= ", MSE)
  return w
  

def closed_form_solution_enhanced(dataset, top_words):
  size = len(dataset)
  X = enhanced_X(dataset,top_words)
  y = vectorY(dataset)

  XTX = np.dot(X.T, X)
  XTy = np.dot(X.T, y)
  w = np.dot(np.linalg.inv(XTX), XTy)
  
  error = np.dot(np.transpose(y-np.dot(X,w)),y-np.dot(X,w))
  MSE = error/size
  print("MSE CLOSED FORM= ", MSE)
  
  return w

# Implement gradient descent:
  
def gradient_descent(data_matrix, target_y, w_initial, b_initial, lr_initial, precision):
    #start = timeit.default_timer()
    size = len(target_y)
    XTranspose = np.transpose(data_matrix)
    XTransposeX = np.dot(XTranspose, data_matrix)
    i=1
    step_b = np.dot(XTranspose, target_y)
    w_previous = w_initial
    while True:
        learning_rate = lr_initial/(1+b_initial*i)
        
        step_a = np.dot(XTransposeX, w_previous)
        step = np.subtract(step_a, step_b)
   
        
        w_current = np.subtract(w_previous,(np.dot((2*learning_rate), step)))
        dw = np.subtract(w_current, w_previous)
        dw_norm = np.linalg.norm(dw)
        i = i + 1
        if (dw_norm < precision):
            error = np.dot(np.transpose(target_y-np.dot(data_matrix,w_previous)),target_y-np.dot(data_matrix,w_previous))
            MSE = error/size
            print("MSE GRADIENT= ", MSE)
            break
        w_previous = w_current
    return w_previous
  
def enhanced_X(dataset, top_words):
  size = len(dataset)
  old = matrixX(dataset, top_words)
  new_column1 = np.zeros((size,1))
  new_column2 = np.zeros((size,1))
  new_column3 = np.zeros((size,1))
  i = 0
  inappropriate_words = ["fuck", "cunt", "bitch", "shit", "fucker", "asshole", "motherfuck", "cock", "bullshit", "prick", "bastard", "idiot", "dick"]

  for datapt in dataset:
    for val in datapt["text"]:
      numwords = len(datapt["text"])
      new_column1[i,0] = numwords
    if any("http" in s for s in datapt["text"]):
      new_column2[i,0] = 1
    if(any(x in inappropriate_words for x in datapt["text"])):
      new_column3[i,0] = 1
    i = i+1  

  partial_resultant = np.hstack((old,new_column1))
  resultant1 = np.hstack((partial_resultant,new_column2))
  resultant2 = np.hstack((resultant1,new_column3))
  return resultant2


if __name__ == "__main__":
  # Splitting the data
  training = data [:10000]
  validation = data [10000:11000]
  testing = data [11000:]

  closed_form_solution(validation, 160)
  
  X = matrixX(validation,160)
  y = vectorY(validation)

  #inital_weights = np.random.rand(164,1)
  inital_weights = np.zeros((164,1))
  gradient_descent(X, y, inital_weights, 1e-3, 7.85e-6, 1e-5)

  closed_form_solution_enhanced(validation, 160)

  