# Face and Fingerprint Recognition System

This is a Flask-based face and fingerprint recognition system that provides user signup and login functionality with anti-spoofing measures. The system uses computer vision techniques to capture and compare user's face and fingerprint data for authentication.

## Features

- **User Signup**
  - Captures multiple face encodings and fingerprint data.
  - Stores temporary user data with a unique token.

- **Complete Signup**
  - Requires user to provide their name, region, and role.
  - Moves the user's data from temporary storage to the main user collection.

- **Login**
  - Captures the user's face for recognition.
  - Compares the face encoding with stored encodings.
  - Captures the user's fingerprint for recognition.
  - Compares the fingerprint data with stored fingerprints.

- **Anti-Spoofing Measures**
  - Texture Analysis (Local Binary Pattern): Checks for the presence of texture typical in a real face.
  - Blink Detection: Detects if the user blinks during capture.
  - Motion Analysis: Analyzes head motion to ensure a live user.
  - Noise Analysis: Ensures the screen has sufficient noise to prevent image spoofing.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

- Python 3.x
- Flask
- OpenCV
- NumPy
- face_recognition
- Flask-PyMongo
- PyFingerprint

Install these dependencies using `pip install -r requirements.txt`.

## Configuration

Ensure that you have MongoDB installed and running. Set the MongoDB URI in the `app.config['MONGO_URI']` variable in the code.

## Usage

1. Start the application by running the script:

   ```
   python app.py
   ```

2. Access the application in your web browser or make HTTP requests to the provided routes.

## API Routes

- `GET /`: Welcome message.

- `POST /signup`: User signup, capturing face and fingerprint data.

- `POST /complete_signup`: Completes the signup process by providing user details.

- `POST /login`: User login, capturing and comparing face and fingerprint data for authentication.

## Error Handling

- The application handles errors using HTTP status codes and provides informative error messages in JSON format.

Feel free to extend and customize this codebase to meet your specific requirements or integrate it with a front-end application for a complete user authentication system.
