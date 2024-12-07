import pandas as pd
import numpy as np

from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split



def prepare_data(file_path, test_size):
    data = pd.read_csv(file_path)

    labels = data['label'].values
    features = data.drop(columns=['label']).values

    return train_test_split(features, labels, test_size=test_size, random_state=42)

def get_model(num_classes):
    model = Sequential([
        Input(shape=(42,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data('keypoint.csv', 0.2)
    model = get_model(4)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    model.save('test_model.keras')
