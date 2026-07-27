from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def create_model(num_classes):

    model = Sequential()

    # Input Layer
    model.add(Input(shape=(64, 64, 3)))

    # First Convolution Block
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )

    # Second Convolution Block
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )

    # Convert feature maps into a vector
    model.add(Flatten())

    # Hidden Layer
    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    # Reduce Overfitting
    model.add(
        Dropout(0.5)
    )

    # Output Layer
    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    return model