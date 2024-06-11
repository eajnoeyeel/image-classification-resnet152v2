# ResNet152V2 Image Recognition

This project demonstrates the training and validation of a ResNet152V2 model for image recognition using TensorFlow and Keras. The model is trained to classify images into eight distinct categories. 

## Project Structure


## Requirements

- Python 3.8
- TensorFlow 2.10
- NumPy
- Matplotlib

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/ResNet152V2-Image-Recognition.git
    cd ResNet152V2-Image-Recognition
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv tf2.10
    source tf2.10/bin/activate  # On Windows use `tf2.10\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Setup CUDA and cuDNN:**

    Ensure that you have the correct versions of CUDA and cuDNN installed. Refer to the official [TensorFlow documentation](https://www.tensorflow.org/install/gpu) for detailed instructions.

## Training the Model

1. **Prepare the dataset:**

    Ensure your dataset is organized as shown in the project structure. The dataset should be divided into training and validation sets.

2. **Run the Jupyter Notebook:**

    Launch Jupyter Notebook and open `ResNet152V2.ipynb`:

    ```sh
    jupyter notebook ResNet152V2.ipynb
    ```

3. **Train the model:**

    Follow the steps in the notebook to train the model. The model will be saved in the `bin` directory after training.

## Usage

1. **Inference:**

    Load the trained model and use it to make predictions on new images:

    ```python
    from tensorflow.keras.models import load_model
    model = load_model('bin/resnet152v2_class8.keras')

    # Use the model to predict new images
    ```

## Results

- The trained model achieves high accuracy on the validation dataset.
- Model architecture and training process are visualized in the notebook.

## Contributions

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License.

