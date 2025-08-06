#  DeepFace Face Verification with ArcFace

This project performs **face verification** using the [DeepFace](https://github.com/serengil/deepface) library with the **ArcFace** model.

It verifies whether two input images belong to the same person, while applying filters like:
-  Rejecting anime or non-human faces
-  Rejecting images with multiple faces
-  Flagging low-confidence matches for human review

---

## Features

- Uses **ArcFace** for high-accuracy face embedding comparison
- Face detection with **RetinaFace**
- Automatically:
  - Rejects anime or invalid faces
  - Rejects images with more than one face
  - Flags cases where similarity is low (`distance > 0.4`)

---

## Requirements

- **Python 3.9.13**  
  *(TensorFlow support ends at 3.12 as of now)*

- Virtual Environment with:
  - `deepface`
  - `tensorflow`
  - `tf-keras`

You can install dependencies like this (after activating the venv):

```bash
pip install deepface
pip install tensorflow
pip install tf-keras
```

---

##  Environment Setup

1. **Install Python 3.9.13**  
   [Download from python.org](https://www.python.org/downloads/release/python-3913/)

2. **Create a virtual environment** inside your project folder:

   ```bash
   python -m venv venv
   ```

3. **Activate the environment**:

   ```bash
   venv\Scripts\activate
   ```

4. **Install required packages** (see above).

---

##  Manual Weights Download

Due to firewall restrictions, the following model weights were downloaded manually:

- `arcface_weights.h5`
- `retinaface_resnet50.h5`

 Place them in the following folder:

```
C:\Users\YourName\.deepface\weights\
```

> Replace `YourName` with your actual Windows username.

---

##  How It Works

### Input:
- Two image files (e.g., `img1.jpg`, `img2.jpg`)

### Rules:
1. **Reject** if either image contains no face or more than one face.
2. **Reject** if DeepFace cannot detect a valid human face (likely anime/cartoon).
3. **Flag for human review** if face distance > 0.4.
4. **Accept** only if valid, single human face and distance ≤ 0.4.

---

## Example Output

```
 Valid face
 Valid face

Verification result:
Match: True
Distance: 0.323
Verified successfully (distance acceptable).
```

Or:

```
Rejected: images/ilias.jpg contains 2 faces.
One or both images were rejected. Stopping.
```

---

## Project Structure

```
Face Verification/
│
├── venv/                       # Virtual environment
├── images/
│   ├── img1.jpg
│   └── img2.jpg
├── Faceverification.py        # Main script
└── README.md
```

---

## Credits

- [DeepFace](https://github.com/serengil/deepface)
- [RetinaFace](https://github.com/serengil/retinaface)
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)

---


