# In-Depth Exploration of Image Classification with CNNs and Transfer Learning

This repository houses an extensive project focused on the nuanced application of Convolutional Neural Networks (CNNs) and Transfer Learning for the task of Image Classification. The project is structured to provide a comprehensive understanding of modern deep learning techniques applied to image data, leveraging both custom-built models and pre-trained architectures to address various challenges in image classification 

## Project Description

The project includes several tasks:

1. **Transfer Learning:** The project delves deep into the realm of Transfer Learning by fine-tuning a sophisticated pre-trained ResNet-50 model on the Oxford Pet Dataset. This segment is designed to showcase the nuanced approach of adapting a model, trained on a generic task, to specialize in a new, more specific task. The process involves a comparative analysis between fully retraining the model versus fine-tuning only the final layers, providing insights into the trade-offs and benefits of each method in terms of performance and computational efficiency.
2. **Training a CNN from Scratch:** Beyond leveraging existing models, the project also explores the construction and training of a bespoke CNN tailored for the CIFAR-10 dataset. This custom model incorporates three convolutional layers, each followed by activation using the innovative Mish function, showcasing the model's ability to learn from scratch.

3. **Adding Batch Normalization:**  Building on the custom CNN, this phase integrates batch normalization techniques to investigate their impact on stabilizing and accelerating the training process. This exploration not only enhances the model's performance but also provides a deeper understanding of how normalization influences learning dynamics.

4. **Optimizing the CNN Architecture:**  In pursuit of optimal performance, this section is dedicated to refining the CNN's architecture. Through a series of experiments, the project evaluates the effects of varying filter sizes, integrating cutting-edge normalization functions such as EvoNorm, and employing advanced optimization strategies like AdamW. This iterative process of tweaking and testing different architectural elements is crucial for pushing the boundaries of what the model can achieve.

Throughout the project, a strong emphasis is placed on empirical evaluation and analysis. This includes detailed reporting of training and validation metrics, such as loss curves and accuracy charts, for both the ResNet-50 adaptation and the custom CNN models. Additionally, the project offers an in-depth error analysis, identifying common misclassifications and discussing potential strategies for improvement. 

## Technologies Used

The project was implemented using Python and the following libraries and frameworks:

- **PyTorch:** At the core of the project's implementation is PyTorch, a leading deep learning library that offers dynamic computation graphs and a rich ecosystem of tools and libraries for building complex neural network architectures.    
- **PyTorch Lightning:**
- **fastai:** To streamline the training process and reduce boilerplate code, PyTorch Lightning is used, enhancing code readability and maintainability.
- **Matplotlib:** This high-level library built on top of PyTorch is utilized for its efficient training utilities and pre-built models, aiding in rapid experimentation and prototyping.
- **NumPy:** For data visualization and numerical operations, Matplotlib and NumPy are employed, providing the necessary tools for plotting detailed graphs and performing high-level mathematical computations.
- **Weights & Biases:** To track experiments, log performance metrics, and visualize results in real-time, the project integrates Weights & Biases, a powerful tool for experiment tracking and model optimization.

