import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Harmonisierung der Daten
def harmonize_data(genomic_data, proteomic_data):
    scaler = StandardScaler()
    genomic_data = scaler.fit_transform(genomic_data)
    proteomic_data = scaler.fit_transform(proteomic_data)
    combined_data = np.concatenate((genomic_data, proteomic_data), axis=1)
    return combined_data

# 2. Modellarchitektur
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Hauptworkflow
def main():
    # Beispielhafte Daten
    genomic_data = np.random.rand(100, 10)  # 100 Proben, 10 genomische Merkmale
    proteomic_data = np.random.rand(100, 5)  # 100 Proben, 5 proteomische Merkmale
    labels = np.random.randint(0, 2, 100)  # 0: Gesund, 1: Krank

    # Daten harmonisieren
    data = harmonize_data(genomic_data, proteomic_data)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Modellaufbau und Training
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

    print("Training abgeschlossen!")

if __name__ == "__main__":
    main()
