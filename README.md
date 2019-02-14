# LSST WP 3.2: Milky Way
## Task 3.2.1 - Hierarchical Star/Galaxy Classifier

***
***

### Authors:
Sergey Koposov  
Leigh Smith  
Vasily Belokurov  
Wyn Evans  

### Requirements:

#### System Packages
python3  
python3-dev  
python3-tk  
source extractor - https://www.astromatic.net/software/sextractor  

#### Python Packages
sewpy - https://sewpy.readthedocs.io/en/latest/index.html  
see requirements.txt  


### Instructions:
code_builder.py - compiles the c-code  
tester.py - generates an image, extracts sources, fits them and returns a pandas dataframe with log likelihoods  


***
***

## Status of Deliverables

### D3.2.1.1:
Written proposal for Level 3 star-galaxy classifier delivered to Science Collaboration
###### _Completed_

### D3.2.1.2:
A set of priors on the stellar distribution in the Galaxy and galaxy luminosity functions to the Science Collaboration.
###### _In progress_

### D3.2.1.3:
A test suite for the classifier to assess performance at producing properly calibrated probabilities, as a function of position and other survey conditions.
###### _In progress_

### D3.2.1.4
Incorporation of priors to generate a more robust star-galaxy classification, results assessed using test suite.
###### _In progress_
