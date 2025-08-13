from deepface import DeepFace
from enum import Enum
from PIL import Image, ImageOps
import numpy as np

# Image paths
img1_path = "images/ilias.jpg"
img2_path = "images/ilias1.jpg"

class Statuses(Enum):
    REVIEW = "review"
    PASS = "pass"
    REJECT = "reject"

def preprocess_image_with_rotation(image_path, angles=[0, 90, 180, 270]):
    """Try rotating the image until a face is found."""
    img = ImageOps.exif_transpose(Image.open(image_path).convert('RGB'))
    
    for angle in angles:
        rotated = img.rotate(angle, expand=True)
        img_np = np.array(rotated)
        try:
            DeepFace.extract_faces(img_np, enforce_detection=True)
            print(f"✅ Face detected at {angle}° for {image_path}")
            return img_np  # Return image as numpy array
        except:
            continue
    
    print(f"❌ No face detected at any angle for {image_path}")
    return None


def is_valid_face(img_path):
    """Checks if image has a single detectable face (with rotation fix)."""
    img_np = preprocess_image_with_rotation(img_path)
    if img_np is None:
        print(f" Rejected: {img_path} probably contains an anime or invalid face.")
        return False, None, "Anime or no recognizable face"

    # Detect faces (already rotated)
    faces = DeepFace.extract_faces(img_np, enforce_detection=True)

    if len(faces) != 1:
        print(f" Rejected: {img_path} contains {len(faces)} faces.") 
        return False, None, "Multiple or no faces detected."

    return True, img_np, " Valid face"


def verify_faces(img1_np, img2_np):
    models = ["ArcFace", "Facenet"]
    distance_threshold = 0.4
    distances = {}

    try:
        for model in models:
            result = DeepFace.verify(
                img1_path=img1_np,
                img2_path=img2_np,
                model_name=model,
                enforce_detection=True
            )

            verified = result["verified"]
            distance = result["distance"]

            print("\n Verification result:")
            print(f"{model} : ")
            print(f"Match: {verified}")
            print(f"Distance: {distance:.3f}")

            distances[model] = distance

        # Use average distance (or pick your preferred model)
        avg_distance = sum(distances.values()) / len(distances)

        if avg_distance > distance_threshold:
            print(" Flagged: Distance is too high. Needs human review.")
            return Statuses.REVIEW, avg_distance

        return Statuses.PASS, avg_distance

    except Exception as e:
        print(" Rejected during verification. Probably invalid or anime face.")
        return Statuses.REJECT, None


# === MAIN EXECUTION ===

# Validate both images
img1_valid, img1_np, _ = is_valid_face(img1_path)
img2_valid, img2_np, _ = is_valid_face(img2_path)

# Proceed only if both are valid
if img1_valid and img2_valid:
    status, distance = verify_faces(img1_np, img2_np)
    if status == Statuses.REVIEW:
        print(" Action required: Send to human for validation.")
    elif status == Statuses.PASS:
        print(" ✅ Verified successfully (distance acceptable).")
    elif status == Statuses.REJECT:
        print(" ❌ Verification failed.")
else:
    print(" ❌ One or both images were rejected. Stopping.")
