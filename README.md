# 2021_summer_fellows
A repository for the 2021 FRI Energy Analytics summer fellows. Contains code and work from the 2021 FRI Summer Fellows from June 1 to July 25 2021. Includes Orphan Wells, Plug and Abandon, and Transformers teams.

## Organization

### Contents:

    Orphan wells
    Plug and Abandon
    Transformers

### Description

#### Orphan Wells 

This project built a SQL database with Texas Railroad Commission and EIA data. The database was then queried to produce training data to predict when wells would likely be orphaned by companies. We find that several features are useful in predicting which wells are most likely to be orphaned and which companies are most likely to orphan wells. 

#### Plug and Abandon

The state of Texas has thousands of orphaned oil wells that need to be plugged up.This project aims to optimize this process by minimizing the distance travelled visiting the orphaned wells. Using road data from the Texas Railroad Commission (RRC), I created an environment for a reinforcement learner algorithm to work with. I trained a Deep Q Learner (DQN) to minimize the total distance travelled within a single county. I then applied the algorithm to other environments to see how it would generalized. During this project, I learned about shapefiles, reinforcement learning, modularization, and how to work with big data. Those who wish to continue this project should investigate ways to incorporate the well plug rate into the environment and learner in order to account for other financial costs related to plugging these wells. In addition, it may be worthwhile to develop a learner that can minimize the environmental risk of leaving certain wells open.

#### Transformers

We can’t currently predict well log data more than a few feet away from the tool while drilling. Our solution is to create a system that can predict well log data much further with much less initial data or potentially no initial data. To do this we use a set of transformers for the prediction along with an accompanying neural network to pick the best transformer for the current well. This project contains cleaned LAS files from Kansas from 2014-2020, a transformer attention approximation with a fully connected network, and shell options for training. 
