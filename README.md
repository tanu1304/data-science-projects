# data-science-projects
A collection of my Data Science and Python Projects.

# Cross-Domain Caricature Recognition

##  Project Overview

This project focuses on **cross-domain caricature recognition** using deep learning and facial feature embeddings. The objective is to identify the person represented in a caricature by comparing its facial representation with reference embeddings extracted from real facial images.

Caricatures often contain exaggerated facial features and artistic distortions, making identity recognition more challenging than conventional face recognition. This project explores the use of pretrained facial feature representations, data augmentation, embedding classification, and cosine similarity to address this challenge.

## h kya Objectives

* Recognize identities from caricature images using facial embeddings.
* Leverage pretrained deep learning models for facial feature extraction.
* Improve robustness through image augmentation.
* Train an embedding classifier using Cross-Entropy Loss and Center Loss.
* Evaluate recognition performance using Top-1, Top-3, and Top-5 accuracy.
* Explore test-time augmentation to improve prediction consistency.

## Methodology

### 1. Facial Feature Extraction

A pretrained **Inception-ResNet-V1 model**, provided by `facenet-pytorch` and pretrained on the VGGFace2 dataset, is used to extract 512-dimensional facial embeddings.

These embeddings represent facial characteristics in a numerical feature space and are used for identity recognition.

### 2. Data Augmentation

A custom augmentation pipeline is applied to introduce variations in facial appearance and improve robustness.

The transformations include:

* Color jittering
* Random posterization
* Sharpness adjustment
* Random perspective transformation
* Random rotation
* Resizing and tensor conversion

These transformations are designed to simulate some visual variations found in caricatures. They are augmentation operations and do not involve training a GAN.

### 3. Embedding Dataset Preparation

The precomputed real-face, augmented-face, and caricature embeddings are used to prepare the training and evaluation data.

| Parameter             | Value  |
| --------------------- | ------ |
| Training embeddings   | 13,100 |
| Evaluation embeddings | 4,272  |
| Identity classes      | 205    |
| Embedding dimension   | 512    |

These values correspond to the dataset configuration used in the notebook.

### 4. Embedding Classifier

A lightweight neural network is trained on the extracted embeddings.

The classifier consists of:

* Batch Normalization layer with 512 input features
* Fully connected layer with 205 output classes

The classifier produces identity scores for the given facial embeddings.

### 5. Training Strategy

The training objective combines two loss functions:

* **Cross-Entropy Loss:** Used to train the classifier to predict the correct identity.
* **Center Loss:** Used to encourage embeddings belonging to the same identity to remain close to their corresponding class centers.

The combined loss is defined as:

`Total Loss = Cross-Entropy Loss + α × Center Loss`

The notebook uses the following training configuration:

* Optimizer: Adam
* Learning rate: 0.0001
* Batch size: 64
* Number of epochs: 60
* Center Loss weight (α): 0.5

### 6. Cosine Similarity-Based Recognition

For recognition, the notebook computes class-mean reference embeddings from the training embeddings. The test caricature embeddings are compared with these reference embeddings using cosine similarity.

The identity with the highest similarity score is selected as the Top-1 prediction. The Top-3 and Top-5 predictions are also evaluated.

### 7. Test-Time Augmentation

Test-time augmentation (TTA) is used to improve prediction robustness.

For each test caricature, embeddings are extracted from:

* The original image
* The horizontally flipped image

The two embeddings are averaged and L2-normalized before similarity-based identity matching.

## 📊 Results

The notebook reports the following recognition results:

| Evaluation Metric | Accuracy |
| ----------------- | -------: |
| Top-1 Accuracy    |   83.66% |
| Top-3 Accuracy    |   91.92% |
| Top-5 Accuracy    |   94.66% |

The evaluation included 4,272 caricature samples, of which 3,574 were correctly classified at Top-1 and 698 were misclassified.

These results reflect the particular dataset split and experimental configuration used in the notebook.

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* FaceNet (`facenet-pytorch`)
* Inception-ResNet-V1
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## 📁 Project Outputs

The notebook saves several trained-model and evaluation artifacts, including:

* Trained embedding classifier weights
* Center Loss class centers
* Class-mean reference embeddings
* Refined real-face embeddings
* Caricature predictions and ground-truth labels
* Top-3 and Top-5 predictions
* Identity-to-label mapping files
* Classification report
* Confusion matrix CSV
* Final results summary

## How to Run

1. Open the notebook in Google Colab.
2. Ensure that the required dataset and precomputed embedding files are available.
3. Mount Google Drive and update the dataset paths according to your folder structure.
4. Install the required dependencies, including `facenet-pytorch`.
5. Run the notebook cells in sequence to load the embeddings, train the classifier, and evaluate recognition performance.

**Note:** The dataset, precomputed embeddings, and saved model artifacts are not automatically included in this repository. The paths used in the notebook may need to be updated for your own environment.

## Future Improvements

* Evaluate the approach on additional caricature datasets and unseen artistic styles.
* Compare performance with and without Center Loss.
* Explore end-to-end fine-tuning of the feature extractor.
* Investigate more advanced domain adaptation and feature alignment methods.
* Improve generalization across different caricature styles.
* Conduct additional experiments to validate recognition performance.

## 👩‍💻 Author

**Tanu Garg**

* GitHub: [tanu1304](https://github.com/tanu1304)
* Project Notebook: [Cross-Domain Caricature Recognition on Google Colab](https://colab.research.google.com/github/tanu1304/data-science-projects/blob/main/Cross_Domain_Caricature__Recog_final.ipynb)
