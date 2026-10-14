# Federated Cervical Cell Image Classification

This project explores federated learning for cervical cytology image classification. It combines a PyTorch notebook that trains and evaluates a simulated multi-client workflow with a Streamlit app that runs predictions using a saved model.

The motivation is to study collaborative model training in medical research settings where sharing image data can be difficult. The notebook demonstrates this idea by keeping client data partitions local during each training round and combining the resulting model parameters.

## Use Case

The project is intended for researchers and students exploring cervical cell image classification and federated optimization. The notebook provides an experimental training workflow, while the app makes the saved classifier accessible for interactive image predictions.

## Project Workflow

The training notebook, [`notebooks/federated-learning-of-medical-imaging.ipynb`](notebooks/federated-learning-of-medical-imaging.ipynb), implements the following workflow:

1. Loads the SIPaKMeD cervical cell image dataset from Kaggle.
2. Resizes images to 224 x 224 pixels, applies random horizontal and vertical flips, and normalizes them using ImageNet statistics.
3. Creates a random 80/20 training and validation split, then trains an ImageNet-initialized MobileNetV3-Small centralized baseline for three epochs.
4. Divides the training data among three simulated clients. Each client trains locally for three epochs per round, for five rounds in total.
5. Averages the clients' model state dictionaries to form the updated global model and evaluates it on the validation set after each round.

The notebook saves model checkpoints and round-wise accuracy results to its Kaggle working directory.

## Dataset

The project uses SIPaKMeD, a collection of cervical cell images organized into five classes:

- `im_Dyskeratotic`
- `im_Koilocytotic`
- `im_Metaplastic`
- `im_Parabasal`
- `im_Superficial-Intermediate`

The class names describe cell-image categories used by the dataset. The dataset is accessed through the [Kaggle SIPaKMeD dataset page](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed).

## Application

The Streamlit interface in [`app/app.py`](app/app.py) loads [`app/federated_cervical_model.pth`](app/federated_cervical_model.pth) and accepts BMP, JPG, or PNG cell images. It displays the predicted class and model score, adds a class-specific text profile, and provides a downloadable text report. This offers a simple way to interact with the trained image classifier.

## Setup and Run

From the repository root, create a virtual environment and install the application dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the application from the `app` directory so it can locate the model checkpoint:

```powershell
Set-Location app
python -m streamlit run app.py
```

Open the local address printed by Streamlit and upload a cell image to run a prediction.

## Training Notebook

The notebook is designed around Kaggle's dataset environment and uses `kagglehub`, `numpy`, `pandas`, and `matplotlib` in addition to the application's PyTorch dependencies. When running it outside Kaggle, install those packages and set `dataset_path` to the downloaded SIPaKMeD directory containing the five class folders.

## Recorded Results

The notebook's saved outputs report a centralized baseline validation accuracy of **93.62%**. The recorded validation accuracies for federated rounds 1 through 5 are **96.81%**, **96.71%**, **97.71%**, **96.91%**, and **98.21%**, respectively. A separate final evaluation cell records **98.01%**.

## References

1. Plissiti, M. E. et al. “SIPAKMED: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images.” *2018 IEEE International Conference on Image Processing (ICIP)*, 2018. [https://doi.org/10.1109/ICIP.2018.8451588](https://doi.org/10.1109/ICIP.2018.8451588)
2. McMahan, B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, 2017. [PMLR paper](https://proceedings.mlr.press/v54/mcmahan17a.html)
3. Howard, A. et al. “Searching for MobileNetV3.” *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019. [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
4. PyTorch TorchVision. [MobileNetV3 model documentation](https://docs.pytorch.org/vision/stable/models/mobilenetv3.html).