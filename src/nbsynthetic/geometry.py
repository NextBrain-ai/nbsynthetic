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
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from pandas.core.arrays.categorical import CategoricalDtype
import plotly.graph_objects as go


class concentration(object):
    """
    Functions to perform geometric analysis of data based on
    concentration of distances (Beyer et al., Hinnenburg et al., François et al. )
  
   """
    def __init__(self):
        self


    def random_data(self, df: pd.DataFrame, size):
        """
        Generates a random data frame from an input one. 

        Args:
            df(pd.DataFrame):
                input data

        Returns:
            randomdf(pd.DataFrame):
        """
        if type(df) != pd.DataFrame:
                raise TypeError("Input has to be a pandas dataframe with labels")

        randomdf = pd.DataFrame()
        for c in df.columns:
          # For continuous columns
          if df[c].dtype == float or df[c].dtype == int:
            randomdf[c] = np.random.uniform(
                low=df[c].min(), 
                high=df[c].max(), 
                size=(size,)
                )
          # For categorical columns
          elif isinstance(df[c].dtype, CategoricalDtype):
            randomdf[c] = np.random.choice(a=df[c].unique().as_ordered(), size=size)
          else:
            raise TypeError("Input dataframe needs to have only numerical values as floats/ints and/or pandas categorical dtypes.")

        return randomdf 



    def concentration_distances(self, df: pd.DataFrame, newdf: pd.DataFrame, distance='euclidean'):
        """
        
        Relative contrast (Beyer et al.)
        --------------------------------
        The concentration of distances is a widely known efect of the 'curse of dimensionality' 
        in distance-based data science. 
        Theorem 1 (Beyer et. al). also known as relative contrast: 

                    If  lim_(𝑑→∞) 𝑉𝑎𝑟(‖𝐗𝐝‖/𝐸[‖𝐗𝐝‖]) = 0 , then (𝐷𝑚𝑎𝑥−𝐷𝑚𝑖𝑛)/𝐷𝑚𝑖𝑛 → 0

        The result of the theorem shows that the difference between the maximum and minimum distances 
        to a given query point does not increase as fast as the nearest distance to any point in high 
        dimensional space. This makes a proximity query meaningless and unstable because there is poor
        discrimination between the nearest and furthest neighbor.


        Contrast (Hinnenburg et al.)
        ----------------------------
        Let F be an arbitrary distribution of two points and the distance function  ‖.‖  be an  L𝑘  metric. Then,

                                lim 𝑑→+∞ 𝐸=[(𝐷𝑚𝑎𝑥(𝑑,𝑘)−𝐷𝑚𝑖𝑛(𝑑,𝑘))/𝑑^{1/𝑘−1/2}]=𝐶𝑘 

        where  𝐶𝑘  is some constant dependent on the norm, 𝑘, and  (𝐷𝑚𝑎𝑥(𝑑,𝑘)−𝐷𝑚𝑖𝑛(𝑑,𝑘))/𝑑^{1/𝑘−1/2}  is the 
        relative contrast, 𝜁L𝑘 . Then the metric  𝐷𝑚𝑎𝑥−𝐷𝑚𝑖𝑛  - also called contrast-, will converge at 𝐶𝑘  
        when increrasing the dimensionality  𝑑→+∞  for the euclidean norm ( 𝑘=2 ). 
        It illustrates the concentration phenomenon described by Beyer et al.


        Relative Variance of the norm RV (François et al.)
        --------------------------------------------------

                                 If  lim 𝑑→∞ 𝑉𝑎𝑟(‖𝐗‖𝑝/√𝐸(‖𝐗‖𝑝)=0 

        where the term  𝑉𝑎𝑟(‖𝐗‖𝑝)√𝐸(‖𝐗‖𝑝) is called Relative Variance and written as RV.Intuitively, 
        we can see that RV measures the concentration by relating a measure of spread (standard deviation) 
        to a measure of location (expectation). In that sense, it is similar to the relative contrast that 
        also relates a measure of spread (range) to a measure of location (minimum). As a consequence, 
        high-dimensional data that present high correlation or dependencies between variables will be much 
        less concentrated than if all variables are independent.
        In Beyer et al. , 1999, Hinneburg et al. , 2000 and Aggarwal et al. , 2002, it is implicitly assumed 
        that stability of nearest neighbours search and concentration (i.e. small relative contrast) are 
        linked together. The rationale is in a sense that a distance measure that is highly concentrated 
        brings very little relevant discriminative information (and consequently makes the search for 
        nearest neighbours unstable).


        References
        ----------
        -   Aggarwal, Charu & Hinneburg, Alexander & Keim, Daniel. (2002). On the Surprising Behavior of 
            Distance Metric in High-Dimensional Space. First publ. in: Database theory, ICDT 200, 
            8th International Conference, London, UK, January 4 - 6, 2001 / Jan Van den Bussche ... (eds.). 
            Berlin: Springer, 2001, pp. 420-434 (=Lecture notes in computer science ; 1973).
        -   K. S. Beyer, J. Goldstein, R. Ramakrishnan, and U. Shaft. (1999). When is "nearest neighbor" 
            meaningful? in Proc. 7th Int. Conf. Database Theory, pp. 217–235.
        -   Alexander Hinneburg, Charu C. Aggarwal, and Daniel A. Keim. (2000). What Is the Nearest 
            Neighbor in High Dimensional Spaces? In Proceedings of the 26th International Conference on 
            Very Large Data Bases (VLDB '00). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 506–515.
        -   François, D., Wertz, V., & Verleysen, M. (2007). The Concentration of Fractional Distances. 
            IEEE Transactions on Knowledge and Data Engineering, 19, 873-886.

        Args:
            df(pd.DataFrame):
                original input data
            newdf(pd.DataFrame):
                synthetic data
            distance(string) = 'euclidean'

        Returns:
            randomdf(pd.DataFrame):
        """
        
        if type(df) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        if type(newdf) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        dfmin = min(
        pdist(np.asarray(df), distance)[
            np.where(pdist(np.asarray(df), distance) > 0)
            ]
        )
        dfmax = max(pdist(np.asarray(df), distance))

        newdfmin = min(
            pdist(np.asarray(newdf), distance)[
                np.where(pdist(np.asarray(newdf), distance) > 0)
                ]
        )
        newdfmax = max(pdist(np.asarray(newdf), distance))

        randomdf = self.random_data(df, size = len(newdf))

        randomdfmin = min(
            pdist(np.asarray(randomdf), distance)[
                np.where(pdist(np.asarray(randomdf), distance) > 0)
                ]
        )
        randomdfmax = max(pdist(np.asarray(randomdf), distance))
        norm = (dfmax-dfmin)/dfmin

        print(f"""
            Average distance of points in data with euclidean distance (L2 norm)
            ---------------------------------------------------------------------
            Avg. distance in original data: {pdist(np.asarray(df), 'euclidean').mean():.2f} 
            Avg. distance in synthetic data : {pdist(np.asarray(newdf), 'euclidean').mean():.2f}
            Avg. distance in random data : {pdist(np.asarray(randomdf), 'euclidean').mean():.2f}

            Distances Variance in original data: {pdist(np.asarray(df), 'euclidean').var():.2f} 
            Distances Variance in synthetic data : {pdist(np.asarray(newdf), 'euclidean').var():.2f}
            Distances Variance in random data : {pdist(np.asarray(randomdf), 'euclidean').var():.2f}


            Contrast (Hinnenburg et al.)
            ----------------------------
            Contrast converges to a constant when the dimension
            increases and when the euclidean distance is used.
            If this contrast decreases, as it is the case for 
            Minkowski norms with p > 2,precision can be lost.

            Contrast, (L norm 2), original data = {(dfmax-dfmin):.2f}
            Contrast, (L norm 2), synthetic data = {(newdfmax-newdfmin):.2f}
            Contrast, (L norm 2), random data = {(randomdfmax-randomdfmin):.2f}


            Relative contrast (\u03B6) (Beyer et al.)
            ------------------------------------
            \u03B6 (L norm 2), original data = {(dfmax-dfmin)/dfmin:.2f}
            \u03B6 (L norm 2), synthetic data = {(newdfmax-newdfmin)/newdfmin:.2f}
            \u03B6 (L norm 2), random data = {(randomdfmax-randomdfmin)/randomdfmin:.2f}


            Relative Variance of the norm RV (François et al.)
            --------------------------------------------------
            Its a measure of concentration proposed for François et al.

            Relative variance original data, RV = {np.square(np.asarray(df).var())/np.asarray(df).mean():.2f}
            Relative variance syntehtic data, RV = {np.square(np.asarray(newdf).var())/np.asarray(newdf).mean():.2f}
            Relative variance random data, RV = {np.square(np.asarray(randomdf).var())/np.asarray(randomdf).mean():.2f}


            Normalized RV
            -------------
            Normalized Relative variance original data, RV = {norm/norm:.3f}
            Normalized Relative variance syntehtic data, RV = {((newdfmax-newdfmin)/newdfmin)/norm:.3f}
            Normalized Relative variance random data, RV = {((randomdfmax-randomdfmin)/randomdfmin)/norm:.3f}

            """)
        return 



    def plot_distances(self, df: pd.DataFrame, newdf: pd.DataFrame, distance='euclidean'):

        """
        Plot distances distributions.

        Args:
            df(pd.DataFrame):
                original input data
            newdf(pd.DataFrame):
                synthetic data
            distance(string) = 'euclidean'

        Returns:
            plotly figure"""

        if type(df) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        if type(newdf) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        randomdf = self.random_data(df, size=len(newdf))
        dfdist = pdist(np.asarray(df), distance)
        newdfdist = pdist(np.asarray(newdf), distance)
        randomdfdist = pdist(np.asarray(randomdf), distance)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dfdist, name='Original data'))
        fig.add_trace(go.Histogram(
            x=np.random.choice(newdfdist, int(newdfdist.shape[0]*0.01)), name='Synthetic')
        )
        fig.add_trace(go.Histogram(
            x=np.random.choice(randomdfdist, int(newdfdist.shape[0]*0.01)), name='Random')
        )

        # Overlay all histograms
        fig.update_layout(
            title = 'Probability distribution of distances', 
            barmode='overlay', 
            plot_bgcolor='white'
            )
        # Reduce opacity to see all histograms
        fig.update_traces(opacity=0.45)
        return fig.show()



    def variance_concentration(self, df: pd.DataFrame, newdf: pd.DataFrame):
        """
        Calculate variance concentration of differente datasets (original and synthetic). 
        Also generatesa random dataset with the same size as synthetic. 

        Variance concentration ratios (VCR) is a rigorous and explainable metric to 
        quantify data. To better examine the explainability of manifold learning in 
        terms of DLPs on high-dimensional and low-dimensional data, we adopt a variance 
        concentration ratio (VCR), which was initially proposed by Han et al. 2021 
        to measure high-frequency trading data, to quantify high and low-dimensional 
        data. The VCR is defined as the ratio between the largest singular value of 
        the dataset and the total sum of all singular values. It answers the question of
        wich is the data variance percentage concentrated on the direction of the first 
        singular value.

        References:
        -   Han, Henry & Teng, Jie & Xia, Junruo & Wang, Yunhan & Guo, Zihao & Li, Deqing. (2021). 
        Predict high-frequency trading marker via manifold learning. Knowledge-Based Systems. 
        213. 106662. 10.1016/j.knosys.2020.106662.

        Args:
            df(pd.DataFrame):
                original input data
            newdf(pd.DataFrame):
                synthetic data

        Returns:
            plotly figure"""


        if type(df) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        if type(newdf) != pd.DataFrame:
            raise TypeError("Input data has to be a pandas dataframe with labels")

        randomdf = self.random_data(df, size=len(newdf))
        # Every matrix  𝐴 ∈ ℂ𝑚×𝑛  can be “decomposed” into the 
        # product of three matrices: 𝐴 = 𝑈.Σ.𝑉∗
        # The diagonal values in the Σ matrix (Σ𝑖𝑖) are known as the 
        # singular values of the original matrix A
        s = np.linalg.svd(df, compute_uv=False)
        s1= np.linalg.svd(newdf, compute_uv=False)
        s2 = np.linalg.svd(randomdf, compute_uv=False)

        VCR = np.max(s)/np.sum(s)
        VCR1 = np.max(s1)/np.sum(s1)
        VCR2 = np.max(s2)/np.sum(s2)

        print(f"""
            Singular values
            ---------------
            Singular values for original dataset : {np.round(s/(np.max(s)),3).tolist()}
            Singular values for synthetic dataset : {np.round(s1/(np.max(s1)),3).tolist()}
            Singular values for random dataset : {np.round(s2/(np.max(s2)),3).tolist()}


            Variance concentration ratio (VCR)
            ----------------------------------
            Variance concentration ratio original data = {VCR*100:.2f}%
            Variance concentration ratio synthetic data = {VCR1*100:.2f}%
            Variance concentration ratio random data = {VCR2*100:.2f}%
            """)
        return 

