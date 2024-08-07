Detecting the hand gesture of the target person among a group of people showing various hand gestures.

Repository content:
├── datasets
│   ├── SIGGI_small           # Small subset of images from the SIGGI dataset for program presentation
│   ├── ...                   # Directory containing all datasets (MIAP, HaGRID, SIGGI), totaling 271GB hence not uploaded to GitHub
│
├── datasets_prep
│   ├── ...                   # Code for dataset conversions, previews, tests, etc.
│
├── training
│   ├── MIAP_person_detection.ipynb        # Training progress of the human detector on MIAP dataset
│   ├── SIGGI_target_detection.ipynb       # Progress of each target detector training
│   ├── HaGRID_gesture_recognition.ipynb   # Gesture recognition training progress
│   ├── experiments.ipynb                  # All hyperparameter experiments runs
│   ├── ...                                # Folders and files needed for training, including the original YOLOv7 repo + folders where training progress and results were saved
│
├── evaluation
│   ├── SIGGI_full.ipynb                   # Calculating results based on the full program runs
│   ├── SIGGI_targets.ipynb                # Quantitative tests on all 7 target detectors
│   ├── MIAP_humans.ipynb                  # Quantitative tests on the human detector
│   ├── HaGRID_SIGGI_gestures.ipynb        # Quantitative and qualitative tests of gesture recognition
│   ├── ...                                # Other evaluation results and calculations
│
├── Person-Tailored Gesture Classification
│   ├── modules
│   │   ├── target_person_detection.py
│   │   ├── crop_target_box.py
│   │   ├── gesture_recognition.py
│   │   ├── gesture_synthesis.py   
│   ├── main.py                            # Entry file that integrates all modules
│   ├── run.ipynb                          # Notebook to run the main file with appropriate arguments
│   ├── requirements.txt                   # Project dependencies listed
│   ├── ...                                # Code and files needed to run the program and save results
│
├── program.tar.gz                         # Compressed 'Person-Tailored Gesture Classification' folder for easier download
├── ...                                    # Other files and folders for project management, such as virtual environments, Git LFS, not uploaded to GitHub

To run the full gesture classification pipeline for the selected target please follow steps:
1.
2.
3.
