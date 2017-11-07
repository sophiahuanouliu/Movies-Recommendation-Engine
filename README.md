# Movies-Recommendation-Engine
This repository aims at creating a movies recommendation engine from MovieLens rating dataset. This data set was collected over various periods of time. The size is 190MB, including 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags. Released 4/2015; updated 10/2016 to update links.csv and add tag genome data.

Recommendation systems changed the way inanimate websites communicate with their users. Rather than providing a static experience in which users search for and potentially watch movies, listen to music, and buy products, recommender systems increase interaction with users to provide a richer experience. For example, YouTube’s recommendation system is one of the most sophisticated and heavily used recommendation systems based on neural network algorithm in industry. In addition, recommender systems identify recommendations autonomously for individual users based on past searches and purchases, and also on other users' behavior. 

With the dataset provided by GroupLens Research, we aim to build a recommendation system. The datasets are divided into three main groupings, one with 100K observations another with 1MM observations and finally one with 10 MM observations. This allows comparing computing resources within the Big data space as our model scales upwards from 100K observations to 10 MM observations.

We anticipate the recommendation system to be a web application with a simple front-end user interface, while the backend will contain our machine learning model. Beyond testing the recommendation model with historical data from the movie-lens dataset, we would like to compare the evaluation with the feedback from human users. 

Recommendation systems usually contain 3 types: popularity-based engines: usually the most simple to implement be also the most impersonal
                                                content-based engines: the recommendations are based on the description of the products
                                                collaborative filtering engines: records from various users provide recommendations based                                                 on user similarities
