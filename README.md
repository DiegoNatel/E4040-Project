[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WG_g337P)
# E4040 2025 Fall Project
## TODO: Your Project Title

Repository for E4040 2025 Fall Project
  - Distributed as Github repository and shared via Github Classroom
  - Contains only `README.md` file

## Project Instructions
Please read the project instructions carefully. In particular, pay extra attention to the following sections in the project instructions:
 - [Obligatory Github project updates](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.8ga1w2quwf7y)
 - [Student Contributions to the Project](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.3jlnclcqaru7)

The project instructions can be found here:
https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0 

## TODO: This repository is to be used for final project development and documentation, by a group of students
  - Students must have at least one main Jupyter Notebook, and a number of python files in a number of directories and subdirectories such as `utils` or similar, as demonstrated in the assignments
  - The content of this `README.md` should be changed to describe the actual project
  - The organization of the directories has to be meaningful

## Detailed instructions how to submit this project:
1. The project will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name
3. **Students must rename the repository per the instructions below**
5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud
6. If some model is too large to be uploaded to Github - 1) create google (liondrive) directory; 2) upload the model and grant access to e4040TAs@columbia.edu; 3) attach the link in the report and this `README.md`
7. Submit the report as a PDF in the root of this Github repository
8. Also submit the report as a PDF in Courseworks
9. All contents must be submitted to Gradescope for final grading

## TODO: (Re)naming of a project repository shared by multiple students
Students must use a 4-letter groupID, the same one that was chosen in the class spreadsheet in Google Drive: 
* Template: e4040-2025Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2025Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments.

```
.
├── data/                   # Dataset storage
├── task1/                  # Task 1: LSUN Bedroom Generation
│   ├── lsun_training.ipynb
│   └── results/            # Experimental results for Task 1
├── task2/                  # Task 2: CelebA Face Generation
│   ├── celeba_training.ipynb
│   └── results/            # Experimental results for Task 2
├── task3/                  # Task 3: ImageNet Feature Extraction
│   ├── imagenet_feature_extraction.ipynb
│   └── results/            # Experimental results for Task 3
├── data_utils.py           # Data loading and preprocessing utilities
├── model.py                # DCGAN model architecture (Generator & Discriminator)
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
