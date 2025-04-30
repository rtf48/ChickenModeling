from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

rf_model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42
)

lr_model = LinearRegression()

nn_model = MLPRegressor(hidden_layer_sizes=(64, 32),  # Two hidden layers
                        activation='relu',            # ReLU activation
                        solver='adam',                # Optimizer
                        max_iter=100000,
                        early_stopping=True,        # Enables early stopping
                        validation_fraction=0.1,    # Fraction of training data for validation
                        n_iter_no_change=10,        # Stop if no improvement after 10 epochs
                        random_state=42
                        )

gb_model = GradientBoostingRegressor(
        n_estimators=1000,        # Number of boosting stages
        learning_rate=0.1,       # Learning rate (shrinkage)
        max_depth=5,             # Maximum depth of each tree
        subsample=0.8,           # Fraction of samples used for fitting each tree
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=0.0001,
        random_state=42
    )

svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)