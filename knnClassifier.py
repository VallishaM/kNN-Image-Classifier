class knnClassifier:
  def __init__(self, PATH, K, TEST):
    """
      PATH - path to dataset
      K - the K in kNN
      TEST - the itended ratio of number of test maes to total number of images
    """
    self.K=K
    classes=os.listdir(PATH)
    dataset=[]
    for c in classes:
      for f in os.listdir(PATH+c):
        img = Image.open(PATH+c+"/"+f)
        im2 = ImageOps.grayscale(img)
        dataset.append((c,np.asarray(im2).flatten()))
    random.shuffle(dataset)
    l=len(dataset)
    slice_index=int(l*TEST)
    self.test=dataset[0:slice_index+1]
    self.dataset=dataset[slice_index+1:l]
    
  
  def test_data(self):
    """
    Returns a dictionary with number of correct and incorrect classifications and acuracy over the test data
    """
    correct=0
    incorrect=0
    for ins in self.test:
      maxk=self.inference(ins)
      if maxk==ins[0]:
        correct+=1
      
      else:
        incorrect+=1
    res={}
    res['correct']=correct
    res['incorrect']=incorrect
    res['accuracy']=(correct/(correct+incorrect))
    return res

  def inference(self, ins):
    k=[]
    for train in self.dataset:
      ed=np.linalg.norm(ins[1]-train[1])
      if len(k)<K:
        k.append((ed, train[0]))
        k=sorted(k)
      else:
        for i in range(K):
          if k[i][0]>ed:
            k[i]=(ed, train[0])
            break
    map={}
    for j in k:
      if j[1] in map:
        map[j[1]]+=1
      else:
        map[j[1]]=1
    maxk=0
    maxv=0
    for j in map:
      if map[j]>maxv:
        maxk=j
        maxv=map[j]
    return maxk
  def get_inference(self, path):
    """
      Input : path : path to image to get inference
      Returns predicted class for the image
    """
    img = Image.open(path)
    im2 = ImageOps.grayscale(img)
    return self.inference(('a',np.asarray(im2).flatten()))
