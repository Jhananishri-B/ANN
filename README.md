ANN Notebook — ann.ipynb

Overview

This repository contains a Jupyter notebook `ann.ipynb` that trains a simple artificial neural network (ANN) on the Wine Quality (red) dataset (`winequality-red.csv`). The notebook demonstrates a complete workflow: data loading, preprocessing, class balancing with SMOTE, feature scaling, building and training a Keras Sequential model, and evaluating results with accuracy, precision, recall, F1, and a confusion matrix.

Contents of `ann.ipynb`

- Imports: pandas, numpy, matplotlib, seaborn, scikit-learn utilities, imbalanced-learn (SMOTE), and TensorFlow/Keras.
- Data: loads `winequality-red.csv` and splits features (`X`) and target (`quality`).
- Resampling: uses SMOTE to balance the quality classes.
- Train/test split: stratified split with 80/20.
- Scaling: StandardScaler applied to features.
- Target encoding: converts quality labels to categorical vectors for multi-class classification.
- Model: Keras Sequential network with several Dense layers (256 -> 128 -> 64) and Dropout, softmax output.
- Training: Adam optimizer, categorical crossentropy loss, EarlyStopping on validation loss.
- Plots: training/validation accuracy and loss.
- Evaluation: predictions on test set, accuracy/precision/recall/F1, per-class precision/recall, and a confusion matrix heatmap.

Requirements

The notebook uses the packages listed in `requirements.txt`. Minimum recommended versions (if you want to pin them):

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- tensorflow

Install dependencies (recommended in a virtual environment):

# Using pip (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; pip install imbalanced-learn

Running the notebook

1) Open with Jupyter:
- Launch Jupyter Notebook or JupyterLab in the repository folder and open `ann.ipynb`.

2) Run cells in order:
- The notebook is linear and cells depend on prior definitions. Run top-to-bottom.

3) Notes for reproducibility:
- The notebook uses a random_state=42 for SMOTE and train/test split.
- Training uses EarlyStopping with patience=15; training may stop earlier than the configured epochs.

Converting to a script

If you prefer to run as a script, convert the notebook to a .py file (Jupyter: File > Download as > Python (.py)) and add a guard around the training run:

if __name__ == "__main__":
    # call the training/evaluation function

Dataset

`winequality-red.csv` should be located in the same folder as the notebook. It contains physicochemical properties of red wine samples and a `quality` score (usually 0-10 scale, but dataset specific values).

What you can tune

- Model architecture (units, layers)
- Learning rate (currently 0.0005)
- Batch size and epochs
- SMOTE parameters or different resampling strategies
- Use class-weighting instead of (or in addition to) SMOTE

Troubleshooting

- GPU/CPU: TensorFlow will use available GPU if configured. If you see long training times, consider installing a GPU-enabled TensorFlow build and appropriate drivers.
- imbalanced-learn missing: install with `pip install imbalanced-learn`.
- If plotting fails in non-interactive shells, save figures using `plt.savefig()` instead of `plt.show()`.

License

This repository has no specific license attached.
