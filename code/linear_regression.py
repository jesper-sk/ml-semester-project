import numpy as np

def obtain_optimal_weights(X,Y):

    W = np.zeros((60,1))
    steps = len(Y)-len(X)
    for i in range(steps):

        x = X[i,:].reshape(60,1)
        y = Y[i+10+1,:].reshape(6,1)

        w = np.dot(np.linalg.pinv(np.dot(x,np.transpose(x))),x)

        for j in range(10):
            for jj in range(6):
                w[j+jj] = w[j+jj]*y[jj]
        W += w

    W = W/steps
    print(W)
    return W
