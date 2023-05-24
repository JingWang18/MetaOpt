import numpy as np
import torch
import scipy as sp
#from scipy.spatial.distance import pdist, cdist
from torch.nn.functional import pdist
from torch import cdist
import pylab as plt
import sys, random, sample

class NIC:

  def __init__(self, k=2, device='cuda'):
    self.K = k            #store K
    self.labels_ = []     #labels for each
    self.scores_ = torch.zeros(k).to(device)  #score for each cluster
    self.device = device

  def fit(self, X, y=None):
    """clusterNIC(samples, K)
      Implementation of the clustering algo from
      A Nonparametric Information Theoretic Clustering Algorithm
      Faivishevsky and Goldberger, ICML 2010
    """
    N = len(X)

    #apply data whitening by mulitplying the data by cov(x)^(-1/2)
    #cov = np.cov(X.T)
    #whitened = np.multiply(X,cov)
    whitened = X

    #initialize random assignment
    torch.manual_seed(0)
    if self.K == 1:
        self.labels_ = torch.zeros(N)
        self.cluster_centers_ = torch.mean(X, 0).reshape(1, -1)
        return
    else:
        self.labels_ = torch.randint(0, self.K-1, (N,))
    #import IPython; IPython.embed()
    #Calculate score:
    sNIC = self.score(whitened, True);

    #do until convergence:
    # Go over the points in a circular manner.
    conCount = 0
    converged = False
    idx = 0
    while not converged:
      # For data point xi calculate scores of all possible reassignments of xi to different clusters.
      jOrig = self.labels_[idx]
      scores = torch.zeros(self.K)
      updates = torch.zeros((self.K,2))
      for j in range(self.K):
        score, jOrigScore, jScore = self.diffScore(whitened, idx, j)
        scores[j] = score
        updates[j, 0] = jOrigScore
        updates[j, 1] = jScore

      # Update current assignment C by choosing label ci that leads to the minimal score.
      jMin = scores.argmin()
      self.labels_[idx] = jMin
      self.scores_[jOrig] = updates[jMin,0]
      self.scores_[jMin] = updates[jMin,1]

      # if a label changed, then no convergence
      if self.labels_[idx] != jOrig:
        conCount = 0
      else:
        conCount += 1
      if conCount >= N:
        #print("... converged.")
        converged = True
        break

      #increment circular
      idx = (idx+1) % N

    #cluster centers
    self.cluster_centers_ = torch.zeros(self.K, X.shape[1]).to(self.device)
    for j in range(self.K):
      pts = X[self.labels_==j]
      self.cluster_centers_[j] = torch.mean(pts, 0)

  def predict(self, X):
    """Predict cluster center for new points X"""
    #for i, xi in enumerate(X):
    #  scores = []
    #  for j in range(self.K):
    #    self.labels_[idx]= j
    pass

  def get_nearest_samples(self, X):
    dist = cdist(self.cluster_centers_, X)
    nearest_samples = torch.argmin(dist, dim=1)
    return nearest_samples

    #import IPython; IPython.embed()
    # compute dist between X and centers
    # find argmax samples
    # return indices

    #weighted_log_prob = self._estimate_log_prob
    #nearest_samples = torch.argmax(weighted_log_prob, dim=0).squeeze(-1)

    ## check overlap
    ##np_nearest_samples = nearest_samples.detach().cpu().numpy()

    #topk = torch.topk(weighted_log_prob, k=weighted_log_prob.shape[1], dim=0).indices.squeeze(-1)

    #nearest_samples = []
    #for i in range(weighted_log_prob.shape[1]):
    #    for idx in topk[:,i]:
    #        if idx not in nearest_samples:
    #            nearest_samples.append(idx)
    #            break
    #return torch.tensor(nearest_samples)


  #calculates full score given current state of self.labels_
  def score(self, X, storeScore=False):
    """Caculates S_nic(cluster_centers_)"""
    score = 0.0
    for j in range(self.K):
      idx = self.labels_ == j
      nj = torch.sum(idx)
      coeff = 1.0 / (nj - 1.0)

      #use pairwise cluster calc
      pts = X[idx]
      dists = torch.log(pdist(pts) + 1e-12)
      distSum = torch.sum(dists)

      #score sum
      score += coeff*distSum
      if storeScore:
        self.scores_[j] = coeff*distSum
    return score

  #calculates score given X, with x_idx in cluster j
  def diffScore(self, X, idx, j):
    """Calculates the score with example idx in cluster j"""
    currJ = self.labels_[idx];
    if currJ == j:
      #for k in range(self.K):
      # print "Score", k, self.scores_[k]
      return torch.sum(self.scores_), self.scores_[j], self.scores_[j]

    #grab point we're considering
    pt = X[idx]

    #put point in limbo/cache score
    self.labels_[idx] = -1
    currJScore = self.scores_[currJ]
    jScore = self.scores_[j]

    #calculate cluster score for currJ without idx
    idxCurrJ = self.labels_==currJ
    ptsCurrJ = X[idxCurrJ]
    nCurrJ = torch.sum(idxCurrJ)+1
    ptDists = cdist(pt.reshape(1, -1), ptsCurrJ)
    ptSum = torch.sum(torch.log(ptDists + 1e-12))
    coeffCurrJ = 1.0 / (nCurrJ - 1.0)
    distSumCurrJ = self.scores_[currJ] / coeffCurrJ;
    try:
        distSumCurrJ -= ptSum
    except:
        import IPython; IPython.embed()
    self.scores_[currJ] = 1.0 / (nCurrJ - 2.0) * distSumCurrJ

    #calculate cluster score for j with idx
    idxj = self.labels_==j
    ptsj = X[idxj]
    nj = torch.sum(idxj)
    ptDists = cdist(pt.reshape(1, -1), ptsj)
    ptSum = torch.sum(torch.log(ptDists + 1e-12))
    coeffj = 1.0/(nj-1.0)
    distSumj = self.scores_[j] / coeffj
    distSumj += ptSum
    self.scores_[j] = 1.0 / (nj) * distSumj

    #sum score before resetting
    toReturn = torch.sum(self.scores_)
    newCurrJ = self.scores_[currJ]
    newj = self.scores_[j]

    #reset point and scores
    self.labels_[idx] = currJ
    self.scores_[currJ] = currJScore
    self.scores_[j] = jScore;

    #return updated score
    return toReturn, newCurrJ, newj

#Gaussian ring and center blob example
if __name__ == "__main__":
  N=2000
  if len(sys.argv) > 1:
    N = int(sys.argv[1])

  doPlot = False
  if len(sys.argv) > 2:
    doPlot = int(sys.argv[2] != 0)

  #gen ring samples
  ring = sample.noisy_ring(N, (0,0), 10.0)
  ringTrue = [1]*N

  #gen norm samples
  means = np.array([0, 0]);
  cov   = np.matrix([[1,0], [0,1]])
  norm  = np.random.multivariate_normal(means, cov, N)
  normTrue = [0]*N

  #combine samps
  X = np.concatenate( (norm, ring) )
  labels_true = np.concatenate( (normTrue,ringTrue) )

  #run NIC clustering algo - need to report score somehow
  nic = NIC()
  nic.fit(X)

  if doPlot:
    #plot results
    cluster0 = X[nic.labels_==0]
    cluster1 = X[nic.labels_==1]
    print(len(cluster0), len(cluster1))
    plt.scatter(cluster0[:,0], cluster0[:,1], marker='^', c='r')
    plt.scatter(cluster1[:,0], cluster1[:,1], marker="^", c="b")
    plt.show()
