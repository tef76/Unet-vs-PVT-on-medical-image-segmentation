# Unet-vs-PVT-on-medical-image-segmentation
L'objectif de mon TER lors de mon master 2 en Science et ingénierie des Données à l'université de Rouen était de comprendre et de comparer un algorithme à convolution et un algorithme transformer sur de la segmentation d'images médicales.

Mon objectif ici est de faire une courte présentation de ce TER et non de montrer le projet dans sa globalité. 

## Dataset:

Pour cela, j'ai travaillé sur le dataset Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS). Il est composé de 500 images échocardiographiques qui ont chacune était segmentée par un expert. D'après le site du dataset 19 % sont de mauvaise qualité, c'est-à-dire qu'elles ne sont pas utilisables pour diagnostiquer un patient. Elles sont quand même utilisées pour l'apprentissage des modèles, car le dataset n'a que peu de données. 

![image](https://user-images.githubusercontent.com/40719576/229733648-98d664b9-5391-434f-aa35-f012ed98d262.png)


## Unet vs PVT:

Pour comparer les deux algorithmes, j'ai séparé les données en 3, 400 pour les données de train, 50 pour les données de validations et 50 pour les données de test.
J'ai redimensionné toutes les images en 224x224 pour réduire le temps de calcul.
La loss utilisée est la CrossEntropy et la métrique et la Dice. 

### Exemple de segmentation du Unet:

![image](https://user-images.githubusercontent.com/40719576/229738153-4be41a1d-a665-4330-b469-25f28a41f00a.png)

### Exemple de segmentation du PVT:

![image](https://user-images.githubusercontent.com/40719576/229738250-4361ae97-e7c3-43e5-afa9-58c7194a04f9.png)

### Récapitultif des deux algorithmes :

![image](https://user-images.githubusercontent.com/40719576/229738563-c9601b1d-4f65-46a0-bb3d-9c0d77f547a9.png)


## Conclusion :

La conclusion de ce projet est que sur le dataset Camus Unet est meilleur que PVT. On peut supposer que ce problème est fortement relié aux formes de l'image et que les convolutions sont potentiellement des meilleurs descripteurs de forme que les transformers. 

Cela ne veut pas dire que les algorithmes à convolution sont meilleurs que les algorithmes transformers sur la segmentation d'image, pour avoir une vraie estimation des capacités des algorithmes sur ce domaine, il faudrait tester différents algorithmes sur différents datasets. 

Il peut aussi être intéressant de tester des algorithmes hybrides qui utilisent à la fois des transformers et des convolutions.

