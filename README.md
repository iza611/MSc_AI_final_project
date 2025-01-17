Detecting the hand gesture of the target person among a group of people. This was a working repository for preparing the datasets, training all the ML models and putting it all together into the 'Person-Tailored Gesture Classification' program. This was then packed into an easily downloadable 'program.tar.gz' file. 

Some of the code used in this project was adapted from https://github.com/WongKinYiu/yolov7. 

# Repository content:
```
├── datasets
│   ├── ...                   # all datasets (MIAP, HaGRID, SIGGI) were places here, 271GB in total, not uploaded to GitHub
│
├── datasets_prep
│   ├── ...                   # Code for dataset conversions, previews, tests, etc.
│
├── training
│   ├── MIAP_person_detection.ipynb        # training human detector on MIAP dataset
│   ├── SIGGI_target_detection.ipynb       # training seven different target detectors on SIGGI
│   ├── HaGRID_gesture_recognition.ipynb   # training gesture recognition on HaGRID
│   ├── experiments.ipynb                  # hyperparameter experiments
│   ├── ...                                # folders and files needed for training, including the original YOLOv7 repo + folders where training progress and results were saved
│
├── evaluation
│   ├── SIGGI_full.ipynb                   # evaluation of the integratedsystem
│   ├── SIGGI_targets.ipynb                # quantitative tests on all 7 target detectors
│   ├── MIAP_humans.ipynb                  # quantitative tests on the human detector
│   ├── HaGRID_SIGGI_gestures.ipynb        # quantitative and qualitative tests on gesture recognition
│   ├── ...                                # other evaluation results, calculations
│
├── Person-Tailored Gesture Classification
│   ├── modules
│   │   ├── target_person_detection.py
│   │   ├── crop_target_box.py
│   │   ├── gesture_recognition.py
│   │   ├── gesture_synthesis.py   
│   ├── main.py                            # entry file integrating all the modules
│   ├── run.ipynb                          # entry notebook to run the main file
│   ├── requirements.txt                   # project dependencies
│   ├── ...                                # code and files needed to run the program and save results
│
├── program.tar.gz                         # compressed 'Person-Tailored Gesture Classification' folder
├── ...                                    # other files and folders for project management, such as virtual environments, Git LFS, not uploaded to GitHub
```

# Installation and Setup Guide
## Download and Unpack the Program

1. **Download**: Save the `program.tar.gz` file to your computer.
2. **Unpack**:
   - **Windows**: Right-click the file, select “Extract All…”, and follow the prompts.
   - **Mac/Linux**: Open Terminal, navigate to the file location, and run:
     ```bash
     tar -xzvf program.tar.gz
     ```

## Install Dependencies

1. **Open Terminal/Command Prompt**:
   - **Windows**: Search for “cmd” and open Command Prompt.
   - **Mac/Linux**: Open Terminal.
2. **Navigate to Folder**: Use `cd` to go to the folder containing `requirements.txt`.
3. **Install**: Run:
   ```bash
   pip install -r requirements.txt

## Open and Configure the Notebook

1. **Start Jupyter Notebook**:
   - Install Jupyter with pip install notebook if needed.
   - Launch it by running jupyter notebook in Terminal/Command Prompt.
2. **Open run.ipynb**: In the Jupyter interface, navigate to and open run.ipynb.
3. **Modify Arguments**: Edit the arguments in the notebook as needed. Refer to main.py for details. Point SOURCE argument to images to use for inference. To download few sample SIGGI images with blurred faces refer to Appendix.

## Run the Notebook

1. **Execute Cells**: Click on each cell and press Shift + Enter to run them sequentially.
