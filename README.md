أكيد! هذا تنسيق جديد ومنظم لملف الـ `README.md` يعرض المحتوى بشكل احترافي وسهل الفهم:

---

# 🧠 Intelligent Retinal Diagnosis and Care Recommendations System

An AI-based system that classifies retinal images (OCT & CFP) and uses a Large Language Model (LLM) to generate personalized healthcare recommendations based on the diagnosis and user profile. It also includes automatic evaluation of recommendation quality for medical relevance and clarity.

---

## 📁 Project Structure

### 🔍 Diagnosis System
| File | Description |
|------|-------------|
| `CFP_FULLCODE2.ipynb` | **Evaluation notebook for CFP** using VGG19 and EfficientNet with two preprocessing pipelines. |
| `OCT_DenseNet201.ipynb` | **Evaluation notebook for OCT** using DenseNet201 model. |
| `OCT_MobileNetV2.ipynb` | **Evaluation notebook for OCT** using MobileNetV2 model. |
| `OCT_VGG19.ipynb` | **Evaluation notebook for OCT** using VGG19 model. |
| `SplitCFPDataset.ipynb` | **Splitting strategy** for CFP dataset into training, validation, and testing sets. |

---

### 🧠 Trained Model Files (External Download)

Due to file size limits, the following trained models are hosted externally:

| Model | Format | Download Link |
|-------|--------|----------------|
| CFP VGG19 | `.keras` | [📎 Download](https://drive.google.com/file/d/1kJ2DMkY9ztSfWscM49tYFZUAYejGVTUx/view?usp=sharing) |
| OCT VGG19 | `.h5` | [📎 Download](https://drive.google.com/file/d/1sr2PuDyXObgB4UZK60ti3O1APZsMFmXz/view?usp=sharing) |
| Image Type Classifier | `.h5` | [📎 Download](https://drive.google.com/file/d/1Fkp2-WwvOp5uEh2ssFMzosdnlgtStAsm/view?usp=sharing) |

---

### 🤖 Recommender System

| File | Description |
|------|-------------|
| `RS&LLM.ipynb` | LLM-based recommender notebook that takes the diagnosis and user responses, builds a tailored prompt, and sends it to **GPT-4 Turbo**. The generated recommendation is evaluated automatically using a second LLM instance to ensure medical accuracy, relevance, and safety. |

---

## 💡 Summary
This system combines deep learning for disease classification with natural language generation for recommendation, aiming to support early diagnosis and improve accessibility to personalized eye care, especially in resource-limited environments.

---

هل تبيني أضيف فقرة "طريقة التشغيل" أو instructions برضو؟
🔗 Download

🤖 Recommender System
RS&LLM.ipynb
Notebook for generating personalized health recommendations using LLM (GPT-4 Turbo), with evaluation module.
