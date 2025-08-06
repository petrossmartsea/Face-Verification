from deepface import DeepFace

# Image paths
img1_path = "images/real1.jpg"
img2_path = "images/real2.jpg"

def is_valid_face(img_path):
    try:
        # Detect faces
        faces = DeepFace.extract_faces(img_path=img_path, enforce_detection=True)

        #reject if faces are more than 1
        if len(faces) != 1:
            print(f" Rejected: {img_path} contains {len(faces)} faces.") 
            return False, "Multiple or no faces detected."

        return True, " Valid face"

    except Exception as e: # Throw exception if face was not found
        print(f" Rejected: {img_path} probably contains an anime or invalid face.")
        return False, "Anime or no recognizable face"

def verify_faces(img1, img2):
    try:
        # Verify if its the same person in the two pics
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="ArcFace",
            enforce_detection=True
        )
        # Results
        verified = result["verified"]
        distance = result["distance"]

        print("\n Verification result:")
        print(f"Match: {verified}")
        print(f"Distance: {distance:.3f}")

        if distance > 0.4:
            print(" Flagged: Distance is too high. Needs human review.")
            return "review", distance

        return "pass", distance

    except Exception as e:
        print(" Rejected during verification. Probably invalid or anime face.")
        return "reject", None

# Validate both images
img1_valid, _ = is_valid_face(img1_path)
img2_valid, _ = is_valid_face(img2_path)

# Proceed only if both are valid
if img1_valid and img2_valid:
    status, distance = verify_faces(img1_path, img2_path)
    if status == "review":
        print(" Action required: Send to human for validation.")
    elif status == "pass":
        print(" Verified successfully (distance acceptable).")
else:
    print(" One or both images were rejected. Stopping.")
