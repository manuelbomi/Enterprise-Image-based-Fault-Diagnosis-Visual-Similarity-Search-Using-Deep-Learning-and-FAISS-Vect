# Enterprise Image Fault Diagnosis & Visual Similarity Search

## Project Summary

#### Manufacturing companies receive defective or faulty material images from customers. Internal engineers manually review returned images, compare them to historical fault libraries, and identify root causes.

#### This project automates fault identification and recommendation:

- Classifies material faults from customer images
- Retrieves visually similar past defect images using FAISS
- Links similar faults to internal knowledgebase / corrective actions

Enabling faster diagnosis, consistent decisions, and knowledge reuse across engineering teams.

---

## Why This Stack?

<ins>VGG16 for Feature Extraction</ins>

- Proven performance on general image tasks

- Pre-trained on ImageNet → generalizable features

- Lightweight vs. modern ViTs → Lower latency for enterprise use

- Replaceable later with EfficientNet, ConvNeXt-V2, or Vision Transformers

---
  

<ins>FAISS for Vector Search</ins>

- Blazing-fast similarity search on 100K+ images

- Designed by Meta AI for production vector systems

- Supports CPU & GPU indexing

- Perfect for enterprise applications where SPEED & SCALE matter.

---

<ins>TensorFlow</ins>

- Stable for enterprise MLOps

- Integrates with TF-Serving, Kubeflow, GCP/AWS ML stacks

  ---

## Use-Case Workflow

- Engineers upload labeled fault images

- VGG16 extracts embeddings → stored in FAISS index

- Train classifier on fault categories

#### New fault arrives → system returns:

- Predicted fault label

- Similar historical images

- Links to internal case resolutions



## Repo structure
```
enterprise-image-fault-diagnosis/
│
├─ data/
│   ├─ raw/                     # customer-returned images (fault samples)
│   ├─ processed/               # preprocessed + augmented images (optional)
│   └─ faiss_index/             # saved FAISS index + label mappings
│
├─ notebooks/
│   ├─ 01_explore_dataset.ipynb
│   ├─ 02_train_vgg16_classifier.ipynb
│   └─ 03_build_faiss_index.ipynb
│
├─ src/
│   ├─ config.py
│   ├─ extract_embeddings.py
│   ├─ build_faiss_index.py
│   ├─ query_similar_images.py
│   └─ train_classifier.py
│
├─ app/
│   └─ streamlit_visual_search.py
│
├─ requirements.txt
├─ README.md
└─ LICENSE
```

### Project Structure Explained

| Folder | Contents |
|--------|----------|
| `data/` | raw, processed images, FAISS index files |
| `notebooks/` | EDA, training, index building notebooks |
| `src/` | Python scripts for embeddings, FAISS, DL model |
| `app/` | Streamlit UI for engineers |
| `requirements.txt` | dependencies |

---

### Core Pipeline Code

#### This repository uses:

- VGG16 to extract image embeddings

- FAISS to build/query vector index
- 

#### The mina cod ei sprovided belwo: 

```python
import numpy as np
import faiss
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import os

# 1. Load VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# 2. Preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 3. Extract features
def extract_features(img_path):
    features = model.predict(load_and_preprocess_image(img_path))
    return features.flatten()

# 4. Load dataset
image_dir = 'path/to/your/images'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

image_features = []
image_names = []

for img_file in image_files:
    image_features.append(extract_features(img_file))
    image_names.append(os.path.basename(img_file))

image_features = np.array(image_features).astype("float32")  # FAISS requires float32

# 5. Create FAISS Index (Cosine similarity ≈ L2 normalized index)
dim = image_features.shape[1]
faiss.normalize_L2(image_features)  # Normalize vectors first

index = faiss.IndexFlatIP(dim)  # Inner product = cosine when normalized
index.add(image_features)       # Add all vectors to FAISS

print("FAISS index built with", index.ntotal, "images.")

# 6. Search function using FAISS
def find_similar_images(query_image_path, top_n=5):
    query_vec = extract_features(query_image_path).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_n+1)  # Query FAISS
    results = []

    for idx, score in zip(indices[0], distances[0]):
        if image_names[idx] != os.path.basename(query_image_path):  # Skip same image
            results.append((image_names[idx], float(score)))
        if len(results) >= top_n:
            break
    
    return results

# Example usage
# similar = find_similar_images("path/to/query.jpg", top_n=3)
# for img, score in similar:
#     print(img, score)

```

