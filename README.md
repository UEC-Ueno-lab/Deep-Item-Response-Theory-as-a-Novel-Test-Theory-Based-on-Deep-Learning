# Deep Item Response Theory as a Novel Test Theory Based on Deep Learning

This is the repository for the code in the paper : Deep Item Response Theory as a Novel Test Theory Based on Deep Learning
Tsutsumi, E.; Kinoshita, R.; Ueno, M. 
Deep Item Response Theory as a Novel Test Theory Based on Deep Learning. Electronics 2021, 10, 1020. 
https://doi.org/10.3390/electronics10091020

If you find this repository useful, please cite
@Article{electronics10091020,
AUTHOR = {Tsutsumi, Emiko and Kinoshita, Ryo and Ueno, Maomi},
TITLE = {Deep Item Response Theory as a Novel Test Theory Based on Deep Learning},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {9},
ARTICLE-NUMBER = {1020},
URL = {https://www.mdpi.com/2079-9292/10/9/1020},
ISSN = {2079-9292}

# abstract
Item Response Theory (IRT) evaluates, on the same scale, examinees who take different tests. It requires the linkage of examinees’ ability scores as estimated from different tests. However, the IRT linkage techniques assume independently random sampling of examinees’ abilities from a standard normal distribution. Because of this assumption, the linkage not only requires much labor to design, but it also has no guarantee of optimality. To resolve that shortcoming, this study proposes a novel IRT based on deep learning, Deep-IRT, which requires no assumption of randomly sampled examinees’ abilities from a distribution. Experiment results demonstrate that Deep-IRT estimates examinees’ abilities more accurately than the traditional IRT does. Moreover, Deep-IRT can express actual examinees’ ability distributions flexibly, not merely following the standard normal distribution assumed for traditional IRT. Furthermore, the results show that Deep-IRT more accurately predicts examinee responses to unknown items from the examinee’s own past response histories than IRT does.

# data format
Each line the number of exercises a student attempted.
All learners work on the test items in the same order.

1,1,1,0,1
1,0,0,1,0
0,0,0,1,1
,
,
,
