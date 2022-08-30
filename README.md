![GAN](docs/images/GAN.jpg)

# **NB synthetic**
A Simple and robust unsupervised synthetic data generator.

License: MIT

Development Status: Beta

Homepage: [https://github.com/NextBrain-ml/tabular-synthetic-data/]

## **Overview**
Next Brain Synthetic is a simple but robust tabular synthetic data generation library. Synthetic data generation is living a golden age in image generation or speech generation applications. Since the introduction of Generative Adversarial Networks by Ian J. Goodfellow in 2014 [^1]., this algorithm have became essential in data generation, overperforming the already existing ones like Variational Autoencoders or Boltzman Machines. GAN are one of the most versatile neural network architectures in use today.
   GANs are made up of two components: generators and discriminators. The generator model produces synthetic samples from random noise collected using a distribution, which are passed to the discriminator, which attempts to differentiate between the two. Both the generator and the discriminator enhance their abilities until the discriminator can no longer tell the difference between actual and synthetic samples. The simultaneous training of generator and discriminator models is inherently unstable[^2]. Since its presentation, several variants of GAN have been created in order to increase both, its stabiliuty and its accuracy. For example, with the introduction of additional parameters as an extra condition, discriminator has an additional help the Discriminators in the classsification between real and synthetic data. This variant is known as Conditional GAN or CGAN and moves the algorithm from the 'unsuperised learning' field to the 'supervised learning' one, by using this additional 'condition'. Another variant of CGAN are the Auxilary Classifier GAN or ACGAN. The list of improived GAN is long and have been succesfully applied in image generarion applications. 

   
## **Why a basic library for synthetic tabular data generation?**

   When it comes to tabular data, the story of GAN have evolved quietly in comparison with image[^3], video[^4] and speech[^5] generation. There are not many libraries to generate synthetic tabular data and are mostly based in conditional GAN architectures[^6][^7][^8]. However, tabular data are by far the most common in the world. Even more, the majority of potential data applications in many industries has to rely in small datasets and, as Data Scinetists use to say, 'poor quality' data. This is the reason why in a data centric approach[^9], the creation of tools for this kind of data is fundamental. For example, we are helping a large hospital in Spain's psychiatric department in a data analysis project. They presented us with a comprehensive research based on data collected over the last ten years. Psychiatric hospitalizations are critical, and this research began with the goal of developing early detection and prevention protocols. We were provided with the results in the form of a spreadsheet with 38 columns and 300 rows. Certainly, that is a small amount of data for any data scientist. It was, however, a challenging effort for them. What are we going to say to them?  Should we tell them that this data is insufficient for using Machine Learning?.Indeed, with this data, the validity of any statistical technique utilized will be put into question. However, this should not be an impediment to maximizing the value from this study by obtaining actionable insights that may be valuable.
   When we have serval dimensions in the original datasets, we have to choose one as an additional 'condition' for our GAN. We will use this dimension or features to condition the generation of the other features. This is certainly practical when we want to use the dataset to solve a supervised learning problem as for example classification or regression. Then we can use our target variable as condition for the GAN.  

## **Unconditional GANs for tabular data**

   As we mentioned, the evolution of GANs has brought interesting ideas as introducing extra information to the discriminator in order to get better accuracy and give more stability to the net. This variant method requires from a target that will condition the generated synthetic data.  If this target data hasn't enough quality (something that's common) we are adding an important bias to our new generated data. Moreover, many datasets has not an only a target feature, because users wants to make predicitions on different features in order to get more insights. So, to condition the synthetic data to a single feature will also add a bias in the generated data. This is why we have decided to use a non conditional GAN or non supervised GAN (also called vanilla GAN) and treat the synthetic data generation as an unsupervised learning problem. Probably the accuracy we get could be improved by condition the GAN with a reliable target variable, but we wanted to provide a versatile tool for a specific target users: small or medium data sets with poor data quality. 
  
## **Statistical tests**
   For checking the output accuracy we have to compare the original input data with the obtained synthetic data. So we have to address the problem of comparing samples from two probability distributions. There are different satatistical tests available for that purpose as the Student's t-test or the Wilcoxon signed-rank test (a nonparametric version of the paired Student’s t-test). For numerical features we also implement the Kolmogorov-Smirnov test. All the above test compartes one-to-one the probability distributions of each feature in the input dataset and the synthetic data (called two samples test or homogeneity problem). But we also needed a test that was able to compare the similarity of both input/outup complete datasets. We have choosen a quite novel approach: the Maximum Mean Discrepancy (MMD) measurement. MMD is a statistical tests to determine if two samples are from different distributions. This test measures the distance between the means of the two samples, mapped into a reproducing kernel Hilbert space (RKHS). Maximum Mean Discrepancy has found numerous applications in machine learning and nonparametric testing [^10][^11]. MMD tests whether distributions p and q are different on the basis of samples drawn from each of them, by finding a smooth function which is large on the points drawn from p, and small on the points from q. The core test statistic is the difference between the mean function values of the two samples; when this is large, the samples are likely from different distributions. An additional reason why we have choosen this test is because it works well with low sample size data.


## **Limitations**
   Unsupervised GANs have been known to be unstable to train and often resulting in generators that produce unrealistic outputs. Some posible solutions have been adopting deep convolutional generative adversarial networks [^12]. But our target with this library are small and medium size datasets so we have desing a network architectura that is robust generating syntehtic dataset up to 5.000 instances. We have tested input dataset with the same number of instances and the newtork is perfectly stable and with an affordable computational cost. The library works with both numerical and categorical inputs. Concerning the data dimnesionality, the library have been tested with datasets up to 200 dimensions. A limitation emerged from the test is when input data is highly dimensional and with only nunerical features. As a general rule, performance is better when we have a dataset with mixed numerical and categorical features.   

# **Requirements**
nbsynthetic has been developed and runs on Python 3.8.

# **Installation**
pip install nbsynthetic

# **How to use it**

## **1. Input data**
  The first step is to load the data wich we will use to fit GAN. We can do it by importing the `nbsynthetic.data.load_data` function an use the parameters name of filename and decimal character. For example `df = input_data(filename, decimal='.')` . Once imported we have to prepare this data considering the following conditions.</br>
- nbsynthetic does not accept string values. We have to encode categorical featires with strings as a numeric array. We can use any of the encoders available in `sklearn.preprocessing`.
- We have to be sure that datatypes are correctly set. Numerical columns has to will have 'float' or 'int' dtypes. Boolean columns will be pandas Categorical type. 
- Drop or replace NaN values
- nbsynthetic does not accept Datetime columns. We have the option to remove it or transform into categorical features. nbsynthetic contain a module that makes this transformation: `data.data_preparation.manage_datetime_columns`, where the arguments are the dataframe and datetime column's name.

nbsyntehtic includes a module that can do all these transformations: `nbsynthetic.data_transformation.SmartBrain`:
- Correcty assigns datatypes and removes id columns
- Remove columns with a high number of NaN values, replaces NaN values when is possible, and discards the rest of the instances where replacement was not possible.
- Finally, this module is able to augment the dataset when possible id dataset lenght is too short. 

An example of how to do make this steps with nbsynthetic library:
   ```
   from data import input_data
   from data_preparation import SmartBrain

   df = input_data('file_name', decimal=',')
   SB = SmartBrain() 
   df = SB.nbPreparation(df)
   ```
  
## **2. Create a GAN instance**
   ```
   from vgan import GAN
   ```

The arguments for the GAN instance are:
- GAN : Vanilla GAN
- df : input data
- samples:number of instances in the synthetic dataset <br/>
We have also additional parameters we can change in the GAN (it's not recomended, by the way).
- initial_lr (default value = 0.0002): Initial learning rate. For more information go [here](https://keras.io/api/optimizers/).
- dropout (default value = 0.5). Droput value. For more information go [here](https://keras.io/api/layers/).
- epochs (default value = 10). Number of epochs. For more information go [here](https://keras.io/api/models/model_training_apis/).

## **3. Generate a synthetic dataset**

   Then, we can directly create a synthetic dataset with the desired number of instances or samples. 
   ```
   from synthetic import synthetic_data

   samples= 2000 #number of samples we want to generate
   newdf = synthetic_data(
       GAN, 
       df, 
       samples = samples
       )
   ```
## **4. Statistical tests**
   The last step is to check how similar is the synthetic dataset with the input one. We will apply several statistical test as explined before. The most important one is the the Maximum Mean Discrepancy test (MMD).
  ```
  from statistical_tests import mmd_rbf, Wilcoxon, Student_t, Kolmogorov_Smirnov
  
  """
  MMD is a statistical tests to determine if two samples 
  are from different distributions. This statistic test 
  measures the distance between the means of the two samples 
  mapped into a reproducing kernel Hilbert space (RKHS).
  Maximum Mean Discrepancy has found numerous applications in 
  machine learning and nonparametric testing.

      Args:

         X: ndarray of shape (n_samples_X, n_features)
         Y: ndarray of shape (n_samples_Y, n_features)
         gamma: float (default:None)

      Returns:
          Maximum Mean Discrepancy (MMD) value: float
   """
   mmd_rbf(df, newdf, gamma=None)
  ```

# **References**
[^1]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.
[^2]: Arjovsky, M., & Bottou, L. (2017). Towards principled methods for training generative adversarial networks. arXiv preprint arXiv:1701.04862.
[^3]: Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8110-8119).
[^4]: Clark, A., Donahue, J., & Simonyan, K. (2019). Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571.
[^5]:Binkowski, M., Donahue, J., Dieleman, S., Clark, A., Elsen, E., Casagrande, N., ... & Simonyan, K. (2019). High fidelity speech synthesis with adversarial networks. arXiv preprint arXiv:1909.11646.
[^6]: Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional gan. Advances in Neural Information Processing Systems, 32.
[^7]: Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
[^8]: Lei Xu LIDS, Kalyan Veeramachaneni. (2018). Synthesizing Tabular Data using Generative Adversarial Networks. arXiv:1811.11264v1
[^9]: Motamedi, M., Sakharnykh, N., & Kaldewey, T. (2021). A data-centric approach for training deep neural networks with less data. arXiv preprint arXiv:2110.03613.
[^10]: Ilya Tolstikhin, Bharath K. Sriperumbudur, and Bernhard Schölkopf (2016). Minimax estimation of maximum mean discrepancy with radial kernels. In Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16). Curran Associates Inc., Red Hook, NY, USA, 1938–1946.
[^11]: A. Gretton, K. M. Borgwardt, M. Rasch, B. Schölkopf, and A. Smola. (2007). A kernel method for the two sample problem. In B. Schölkopf, J. Platt, and T. Hoffman, editors, Advances in Neural Information Processing Systems 19, pages 513–520, Cambridge, MA. MIT Press.
[^12]: Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
