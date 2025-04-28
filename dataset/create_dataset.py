import cv2
import os

def create_dataset(person_name, save_dir="known_faces", num_images=20):
    os.makedirs(f"{save_dir}/{person_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capturing...", frame)

        # Save frame
        cv2.imwrite(f"{save_dir}/{person_name}/img_{count}.jpg", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter your name: ")
    create_dataset(person_name)
