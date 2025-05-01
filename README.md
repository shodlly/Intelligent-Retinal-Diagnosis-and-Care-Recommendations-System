# Intelligent-Retinal-Diagnosis-and-Care-Recommendations-System
An AI system that classifies OCT and CFP retinal images and uses an LLM to generate personalized health recommendations based on diagnosis and user data, with automatic evaluation for safety and relevance.


📁 File Structure & Descriptions
Below is a description of the main files included in this repository, grouped by their purpose:

🔬 Diagnosis System
CFP_FULLCODE2.ipynb
Evaluation notebook for CFP using VGG19 and EfficientNet models with two preprocessing pipelines.

OCT_DenseNet201.ipynb
Evaluation notebook for OCT using DenseNet201 model.

OCT_MobileNetV2.ipynb
Evaluation notebook for OCT using MobileNetV2 model.

OCT_VGG19.ipynb
Evaluation notebook for OCT using VGG19 model.

SplitCFPDataset.ipynb
Splitting strategy for the CFP dataset into training, validation, and testing sets.

🧠 Trained Models
Due to file size, the trained model files are hosted externally and can be downloaded here:

CFP VGG19 model (.keras)
🔗 Download

OCT VGG19 model (.h5)
🔗 Download

Image Type Classifier model (.h5)
🔗 Download

🤖 Recommender System
RS&LLM.ipynb
Notebook for generating personalized health recommendations using LLM (GPT-4 Turbo), with evaluation module.
