import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(self, iterations=50, max_depth=3):
        self.iterations = iterations
        self.max_depth = max_depth
        self.weak_learners = []
        self.multipliers = []
        self.initial_predictions = None

    # Helper function to determine best multiplier to get the lowest result from loss function
    def _find_best_gamma(self, predictions, F, training_labels):
        def loss_function(gamma):
            new_F = np.clip(F + gamma * predictions, 1e-15, 1 - 1e-15)
            loss = -np.sum(
                training_labels * np.log(new_F + 1e-15)
                + (1 - training_labels) * np.log(1 - new_F + 1e-15)
            )
            return loss

        result = minimize_scalar(loss_function, bounds=(0, 1), method="bounded")
        return result.x

    def fit(self, training, training_labels):
        # Initialize predictions with the mean of the target
        self.initial_predictions = np.mean(training_labels)

        F = np.full_like(
            training_labels, fill_value=self.initial_predictions, dtype=float
        )

        for iteration in range(self.iterations):
            # Calculate pseudo-residuals -> calculate the negative gradient of loss function with respect to current model (using log loss)
            # Numpy allows for addition/subtraction/multipication/division on arrays of the same size/shape
            gradients = (training_labels - F) / (F * (1 - F))
            # Clamp gradients to prevent extreme values which could lead to instability/oscillations
            # I found -10 : 10 to be a good range
            gradients = np.clip(gradients, -10, 10)

            # Train new tree (weak learner) using pseudo-residuals
            weak_learner = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=7
            )
            weak_learner.fit(training, gradients)
            self.weak_learners.append(weak_learner)

            # Predict residuals and find the best gamma
            # Find the multiplier that leads to the smallest loss when adding the new weak learner
            predictions = weak_learner.predict(training)
            best_gamma = self._find_best_gamma(predictions, F, training_labels)
            self.multipliers.append(best_gamma)

            # Update the model = Old model + multiplier(new model)
            F += best_gamma * predictions

            # Clamp F to make sure 0 < F < 1
            F = np.clip(F, 1e-15, 1 - 1e-15)

            current_loss = -np.sum(
                training_labels * np.log(F + 1e-15)
                + (1 - training_labels) * np.log(1 - F + 1e-15)
            )
            print(f"Iteration {iteration + 1}, Log Loss: {current_loss:.4f}")

    def predict(self, testing):
        # Create array shaped like testing[0], fill with initial predictions (mean of training labels)
        F_test = np.full(
            testing.shape[0], fill_value=np.mean(self.initial_predictions), dtype=float
        )
        # Iterate through calculated weak learners and gamma values
        for weak_learner, gamma in zip(self.weak_learners, self.multipliers):
            F_test += gamma * weak_learner.predict(testing)

        # Convert probabilities to binary predictions
        return (F_test > 0.5).astype(int)
