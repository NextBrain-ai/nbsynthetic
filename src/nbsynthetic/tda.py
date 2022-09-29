# Author: Javier Marín (javier.marin@softpoint.es)
# Copyright 2022 Softpoint Consultores SL. All Rights Reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================
from operator import add
import math
from numba import jit
from numba.extending import overload
import numpy as np
import scipy
from scipy import stats
from scipy.sparse import issparse
from sklearn.utils import check_array
from ripser import ripser
import plotly.express as px
import plotly.graph_objects as go



class Topology(object):
  """
    Functions to perform topological data analisys:

    Parameters:
    -----------
    X: Input data (np.array)
    Returns:
    --------
    Utils for topological data analysis
  
   """


  def __init__(self):
    self



  def check_array_function(self, X: np.ndarray):
    """Function 'sklearn.utils.validation.check_array."""
    if isinstance(X, np.matrix) or (
        not isinstance(X, np.ndarray) and not sp.issparse(X)
        ):
      X = check_array(
          X,
          accept_sparse=["csr", "csc", "coo"],
          dtype=np.float64,
          copy=copy,
          force_all_finite=force_all_finite,
          ensure_2d=False
          )
      
    return X



  @jit(parallel=True, fastmath=True, forceobj=True)
  def distance_matrix(self, X: np.ndarray):
    """
    Computes the L2 norm on a flattened view of an input array 
    X. L2 is computed on a flattened view of the array: this is
    the square root of the sum of squared elements and can 
    be interpreted as the length of the vector in 
    Euclidean space.
   
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
          An array where each row is a sample and each 
          column is a feature.
    
    Returns
    -------
    matrix : ndarray of shape (n_samples_X, n_samples_X)
        """

    Xnew = self.check_array_function(X)

    matrix = np.linalg.norm(
        add(Xnew[:, None, :],-1 * Xnew[None, :, :]),
        axis=-1
        )
    yield matrix



  def vietory_rips(self, X: np.ndarray):
      """
      For each point cloud in Euclidean space or distance 
      matrix in X, compute the relevant persistence diagram 
      as an array of pairs [b, d]. Each pair represents a 
      persistent topological feature in dimensions H0 and H1
      which is born at b and dies at d.
      
        Parameters:
        -----------
        matrix: ndarray of shape (n_samples_X, n_samples_Y)
          Distance matrix
        Returns:
        --------
        list: 
          Persistence diagram with birth-death pairs[b, d]
          in dimensions H1 and H0 (if available).
        """
      if type(X) != np.ndarray:
        raise TypeError("X has to be a numpy array.")
      
      if X.dtype != float:
        raise TypeError("X has to be an array of floats.")

      matrix = next(self.distance_matrix(X))
      
      if issparse(matrix):
        raise TypeError("Persistence diagrams computation do not support sparse matrices.")
      
      # calculate persistence diagrams with ripser
      # https://ripser.scikit-tda.org/en/latest/
      dgms = ripser(
          matrix, 
          distance_matrix=True,
          )['dgms']
      
      # replace inf. values 
      if len(np.array(dgms, dtype='object')[1]) > 0:
        H0_max = np.max(dgms[0][:-1])
        H1_max = np.max(dgms[1])
        ct = (
            H1_max - np.min(dgms[0][0:,1])
            )/20
        
        if H0_max < H1_max:
          dgms[0] = np.where(
              dgms[0] != np.inf, 
              dgms[0], 
              H1_max + ct
              )
        
        else:
          dgms[0] = np.where(
              dgms[0] != np.inf, 
              dgms[0], 
              H0_max + ct
              )
      
      else:
        dgms[0] = dgms[0][:-1]
      
      return dgms



  def plot_diagram(self, dgms):
    """Plot a single persistence diagram.
    Parameters
    ----------
    diagram : ndarray of shape (n_points, 2)
        The persistence diagram to plot, where the dimensions contains 
        (birth, death) pairs to be used as coordinates in 
        the two-dimensional plot.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.
    
    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the persistence diagram.
    """
    
    if len(np.array(dgms, dtype='object')[1]) > 0:
      H0_max = np.max(dgms[0])
      H1_max = np.max(dgms[1])     
      ct = (H1_max - np.min(dgms[0][0:,1]))/5
       
      if H1_max > H0_max:
        infValues = axisLimit = H1_max 
      
      else:
        infValues = H0_max 
        axisLimit = np.max(dgms[0].T) 
    
    else:  
      infValues = axisLimit = np.max(dgms[0][0:,1].T)*1.1

    fig = go.Figure()
    if len(np.array(dgms, dtype='object')[1]) == 0:
        fig.add_trace(go.Scatter(
          x=dgms[0][0:,0].T, 
          y=dgms[0][0:,1].T, 
          name='H<sub>0',
          mode='markers', 
          marker=dict(
              size=6, 
              color='navy'
              )
          )
      )

    else: 
      fig.add_trace(go.Scatter(
          x=dgms[0][0:,0].T, 
          y=dgms[0][0:,1].T, 
          name='H<sub>0',
          mode='markers', 
          marker=dict(
              size=6, 
              color='navy'
              )
          )
      )  
      fig.add_trace(go.Scatter(
          x=dgms[1][0:,0].T, 
          y=dgms[1][0:,1].T,
          name = 'H<sub>1', 
          mode='markers', 
          marker=dict(
              size=6, 
              color='crimson'
              )
          )
      )
    fig.add_trace(go.Scatter(
        x=np.array([0, axisLimit]),
        y=np.array([0, axisLimit]), 
        mode='lines', 
        line_color='grey', 
        line_width=1, 
        line_dash='dash',
        showlegend=False
        )
    )
    fig.add_trace(go.Scatter(
        x=np.array([0, axisLimit]),
        y=np.array([infValues, infValues]), 
        mode='lines', 
        line_color='grey', 
        line_width=1, 
        line_dash='dash',
        showlegend=False
        )
    )
    fig.update_layout(
        shapes=[
            dict(type="line", xref="x", yref="y",
                  x0=0, y0=0, x1=0, y1=axisLimit, 
                  line_width=1),
            dict(type="line", xref="x", yref='y',
                  x0=0, y0=0, x1=axisLimit, y1=0, 
                  line_width=1)
            ]
    )
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Birth',
        yaxis_title='Death',
        autosize=False, 
        width=500, 
        height=500)

    return fig.show()



  def persistent_entropy(self, dgms, normalize=False):  
      """
      Given a persistence diagram consisting of
      birth-death pairs [b, d], subdiagrams corresponding to
      distinct homology dimensions are considered separately, 
      and their respective persistence entropies are calculated 
      as the (base 2) Shannon entropies of the collections of 
      differences d - b ("lifetimes"), normalized
      by the sum of all such differences.
      
        Parameters:
        -----------
        dgms: ndarray of shape (n_samples, n_features). 
        Persistence diagram with birth-death pairs[b, d]
        
        Returns:
        --------
        python generator with entropy values for H0 and H1
        """
      
      for idx, dgm in enumerate(dgms):
        l = dgm[:, 1] - dgm[:, 0]
        if all(l > 0) and normalize != True:
          L = np.sum(l) 
          p = l / L
          E = -np.sum(p * np.log(p))
          yield E
        
        elif all(l > 0) and Normalize == True:
          E = E / np.log(len(l))
          yield E
        
        elif all(l > 0) != True:
          raise Exception("A bar is born after dying")



  def mann_whitney(self, dgms1, dgms2):
      """
      Mann Whitney test: chack if there are differences in 
      the geometry of both point clouds. If the p-value is 
      smaller than the significance level α, H0 is valid (there 
      are no differences between both geometries).
      
        Parameters:
        -----------
        dgms1: ndarray of shape (n_samples, n_features). 
        Persistence diagram with birth-death pairs[b, d]
        
        dgms2: ndarray of shape (n_samples, n_features). 
        Persistence diagram with birth-death pairs[b, d]
        
        Returns:
        --------
        ndarray of shape (2, ) with p-values values for b and d
        """
      print(f"""
      Mann Whitney U test p-value for dimension HO: {stats.mannwhitneyu(dgms1[0],dgms2[0])[1][0]}
      Mann Whitney U test p-value for dimension H1: {stats.mannwhitneyu(dgms1[0],dgms2[0])[1][1]}
      """)
      return



  def bottleneck(self, dgms1, dgms2):
      """
      Returns the dimension H0 bottleneck distance between two 
      non-empty persistence diagrams. All components are assumed 
      to be born at the beginning of the filtration, 
      that is, all non-trivial points lie in the vertical axis.   
      
      References: 
      Ignacio, P. S., Bulauan, J.-A., & Uminsky, D. (2020).
      Lumáwig: An Efficient Algorithm for Dimension Zero Bottleneck 
      Distance Computation in Topological Data Analysis. 
      Algorithms, 13(11), 291. MDPI AG. Retrieved 
      from http://dx.doi.org/10.3390/a13110291
      
      Parameters
      ----------
      dgms1 : ndarray of shape (n_points, 2)
          Persistence diagram  where the dimensions contains 
          (birth, death) pairs.

      dgms2 : ndarray of shape (n_points, 2)
          Persistence diagram  where the dimensions contains 
          (birth, death) pairs.
             
      Returns
      -------
      d : float
          The bottleneck distance between dgms1 and dgms2.
        
      """
      X = np.asarray(dgms1[0][0:, 1], dtype=object)
      Y = np.asarray(dgms2[0][0:, 1], dtype=object)
      
      if Y.shape[0] < X.shape[0]:
        X, Y = Y, X
      X, Y = X[::-1], Y[::-1]
      
      #Calculate distance
      d, N = 0, X.shape[0] 
      Z = abs(X - Y[0: N]) 
      l = np.argmax(Z)  
      
      # Keep an initial bijection. Systematically modify this bijection to optimize
      # the norm between matched points until the bottleneck matching is recovered.
      dtemp = Z[l]
      
      if N != len(Y) and dtemp < 0.5 * Y[N]:
          d = 0.5 * Y[N]      
      
      elif l >= 0 and len(Z) > 1:
          while len(Z) > 1:
              k = 0.5 * max(X[l], Y[l])
              if np.max(np.delete(Z, l)) <  k and k < dtemp: 
                  d = k
                  break

              elif np.max(np.delete(Z, l)) >= k:
                  if len(Z[Z >= k]) == len(np.where(np.where(Z >= k)[0] >= l)[0]): 
                      d = k
                      break
                  else:
                      X, Y, Z = X[0: l], Y[0: l], Z[0: l]           
                      if len(Z) == 1:
                          d = min(dtemp, k)
                          break
              else:
                  d = dtemp
                  break
       # If one of the persistence diagrams is a singleton  
      else:   
          d = min(dtemp, 0.5 * max(X[0], Y[0]))

      return d