# Import necessary libraries
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the built-in MovieLens 100k or 1 million dataset
# If not already downloaded, it will be automatically fetched
#data = Dataset.load_builtin('ml-100k')
data = Dataset.load_builtin('jester')

# Split the dataset into training and testing sets (75% training, 25% testing)
trainset, testset = train_test_split(data, test_size=0.25)

# Initialize the SVD (Singular Value Decomposition) algorithm for collaborative filtering
algo = SVD()

# Train the model on the training set
algo.fit(trainset)

# Generate predictions for the test set
predictions = algo.test(testset)

# Calculate and print the Root Mean Squared Error (RMSE) of the predictions
rmse = accuracy.rmse(predictions)

# Print a summary of results
print(f"Model Evaluation Completed: RMSE = {rmse:.4f}")
