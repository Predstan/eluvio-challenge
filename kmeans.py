import pandas as pandas
import numpy as np
import numpy.linalg as nla

def dfSimilarity(df,centroids):
    """Calculate similarities for dataframe input
       Implemented Using Matrix Operation
       ||a-b||^2 = |a|^2 + |b|^2 - 2*|a|*|b|
    """

    numPoints = len(df.index)
    numCentroids = len(centroids.index)
    
    
    # norm of points adds a constant bias to distances
    # But calculating helps so that the similarity doesn't go negative
    pointNorms = np.square(nla.norm(df,axis=1))
    pointNorms = np.reshape(pointNorms,[numPoints,1])
    # Calculate the norm of centroids
    centroidNorms = np.square(nla.norm(centroids,axis=1))
    centroidNorms = np.reshape(centroidNorms,(1,numCentroids))
    # Calculate |a|^2 + |b|^2 - 2*|a|*|b|
    similarities = pointNorms + centroidNorms - 2.0*np.dot(df,np.transpose(centroids))
    # Divide by the number of features (32 from embeddings)
    similarities = similarities/32.0
    # clip similaririties so that it doesnt go negative
    similarities = similarities.clip(min=0.0)
    # Square root since it's ||a-b||^2
    similarities = np.sqrt(similarities)
    return similarities

def initCentroids(df,k,feature_cols):
    # Pick 'k' examples at random to serve as initial centroids
    limit = len(df.index)
    centroids_key = np.random.randint(0,limit-1,k) 
    centroids = df.loc[centroids_key,feature_cols].copy(deep=True)
    # the indexes get copied over so we reset them
    centroids.reset_index(drop=True,inplace=True)
    return centroids

def pt2centroid(df,centroids,feature_cols):
    """ Calculate similarities between all points and centroids
         And assign points to the closest centroid + save that distance
    """
    numCentroids = len(centroids.index)
    numExamples = len(df.index)
    # dfSimilarity = Calculate similarities for dataframe input
    dist = dfSimilarity(df.loc[:,feature_cols],centroids.loc[:,feature_cols])
    df.loc[:,'centroid'] = np.argmin(dist,axis=1) # closest centroid
    df.loc[:,'pt2centroid'] = np.min(dist,axis=1) # minimum distance
    return df

def recomputeCentroids(df,centroids,feature_cols):
    """ For every centroid, recompute it as an average of the points
        assigned to it
    """ 
    numCentroids = len(centroids.index)
    for cen in range(numCentroids):
        dfSubset = df.loc[df['centroid'] == cen, feature_cols] # all points for centroid
        if not(dfSubset.empty): # if there are points assigned to the centroid
            clusterAvg = np.sum(dfSubset)/len(dfSubset.index)
            centroids.loc[cen] = clusterAvg
    return centroids

def kmeans(df, k,feature_cols,verbose):
    flag = False
    maxIter = 3000
    iter = 0                      # ensure kmeans doesn't run for ever
    centroids = initCentroids(df,k,feature_cols)
    while not(flag):
        iter += 1
        #Save old mapping of points to centroids for checking congergence
        oldcentroidmap = df['centroid'].copy(deep=True)
        # Perform k-means
        df = pt2centroid(df,centroids,feature_cols)
        centroids = recomputeCentroids(df,centroids,feature_cols)
        # Check convergence by comparing [oldCentroidMap , newCentroidMap]
        newcentroidmap = df['centroid']
        flag = all(oldcentroidmap == newcentroidmap)
        if verbose == 1:
            print("Average distance to centroids:" + str(np.sum(df['pt2centroid'])/len(df["pt2centroid"])))
        if (iter > maxIter):
            print('k-means did not converge! Reached maximum iteration limit of ' \
                + str(maxIter) + '.')
            sys.exit()
            return
    print('k-means converged for ' + str(k) + ' clusters' + \
        ' after ' + str(iter) + ' iterations!')
    return [df,centroids]