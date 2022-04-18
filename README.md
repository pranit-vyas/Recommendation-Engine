# Recommendation-Engine
This is repository for learning and comparing various recommendation engines on MovieLens dataset.

1. A simple pandas based implementation of user-user collaborative filtering on MovieLens dataset
2. Item based collaboration is similar to user based but the user-movie rating matrix is now transposed or flipped sideways. 
It is much faster and more accurate than user based as the number of movies (on which we loop) is smaller than number of users and between two movies number pf common users would be much higher.

Few learnings from the above exercises

- Changing datatypes of columns proves efficient
- With an increase in number of relevant neighbours to predict movie rating, the mse decreases
- Another parameter which can be tweeked is the minimum number of common users between two users to consider them for similarity


3. Pandas and numpy based implementation of Matrix Factorization based recommender system on MovieLens dataset. 
This methodology aims to break down the user-movie rating matrix into two matrices of user-feature (W) and movie-feature matrix (U), such that the predicted rating for user "i" and movie "j" can be computed as w[i] * u[j]
The derivation involves exchanging the predicted rating with the above mentioned dot product and differentiating the loss function i.e. squared error with respect to w and u. Addition of user bias, movie bias and average movie rating is also advisable.
