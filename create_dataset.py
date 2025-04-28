import cv2
import os

def create_dataset(name, save_dir="known_faces", num_images=20):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Start webcam
    cap = cv2.VideoCapture(0)
    print(f"[INFO] Capturing images for {name}... Press 'c' to capture, 'q' to quit.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            img_name = f"{name}_{count}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Saved {img_name}")
            count += 1
            if count >= num_images:
                print(f"[INFO] Collected {num_images} images for {name}.")
                break

        if key & 0xFF == ord('q'):
            print("[INFO] Quitting capture early.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    create_dataset(user_name)
