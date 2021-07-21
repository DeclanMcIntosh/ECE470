# ECE470 Project Code

FULL CODE AT https://github.com/DeclanMcIntosh/ECE470

Today cancer is a leading ailment in society and some of the most common cancers occurring in the gastrointestinal tract. To combat these cancers and perform preventative care colonoscopies are performed to screen for these cancers and remove any abnormalities which may be precursors to these cancers. These precursors are broadly referred to as polyps. Currently, there are large amounts of polyps missed as many as 30\% during these procedures which allows for cancers to grow in a patient.  

To reduce these missed polyps this report details a method to generate a pixel-wise mask of polyps in a colonoscopy image automatically. This "second opinion" from a computer can standardize care and generate a baseline of care between all medical practices. A segmentation formulation of this problem also will enable greater automatic estimation of polyp size and enable better meta-data around polyps for public-health level studies. The proposed method is able to operate in real-time exceeding 15FPS and has a mean Dice Score exceeding 0.85 indicating extremely strong segmentation results. These results are currently limited by the small sizes of datasets available in this domain. 

## Instructions

1. Data can be downloaded from https://drive.google.com/file/d/1aMCd0dbnwLtL3aKZeUHX6JdL5B-yVxbB/view?usp=sharing or from project submission.

2. The dowloanded data can be placed in the main directory and unziped.

3. The trained model and logs can then be extracted in the root directory of the project.

4. Next install Python Version 3.6.6 (This is the only version this code has been implemented for or tested on).

5. Once Python has been installed used PIP to install all dependencies in "dependencies.txt".

6. To demo the program the run "demo.py" this will show you a series of example outputs from the model and the associated RGB inputs and the expected ground truth. All examples are from the test set data. 

