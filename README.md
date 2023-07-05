# Object-Oriented Dynamics Predictor (OODP)

This project is a re-implementation of the Object-Oriented Dynamics Predictor (OODP) originally developed by Zhu et al. ([Guangxiang Zhu*, Jianhao Wang*, Zhizhou Ren*, and Chongjie Zhang, "Object-Oriented Dynamics Learning through Multi-Level Abstraction", Deep RL Workshop NIPS 2018.](https://arxiv.org/abs/1904.07482)) which used Tensorflow 1 [Github project of Zhu et al.](https://github.com/mig-zh/OODP). The goal of OODP is to predict the future dynamics of objects in a scene using deep learning techniques.

This implementation of OODP is based on TensorFlow 2 and was completed as part of the Neural Network Seminar at the University of Hamburg during the **Summer semester of 2020**.

> Warning: **The Game Monsterkong is currently not working because of the missing PygameWrapper lib.**

## Dependencies

The project requires the following dependencies:

- TensorFlow
- NumPy
- tqdm
- logging
- Pillow
- PyGame

These dependencies can be installed using Pipenv. Please refer to the `Pipfile` for specific versions.

## Usage

1. Set up a Python environment with the required dependencies (e.g., using Pipenv).
2. Download and init Monsterkong from the [PyGame-Learning-Environment](https://github.com/ntasfi/PyGame-Learning-Environment/tree/master).
2. Create datasets using the `createDataset.py` script.
3. Train the models using the `train.py` or `train_dist.py` scripts.
4. Use the trained models for prediction and analysis using the `analysis.ipynb` notebook.
5. Refer to the documentation in the `docs/` directory for additional information and resources.

## Results

The results are documented [here](docs/PaperFinal_Korn_Malte_OODP.pdf).

## Acknowledgments

We would like to express our gratitude to Zhu et al. for developing the original Object-Oriented Dynamics Predictor (OODP) and providing the inspiration for this re-implementation. We would also like to thank the Neural Network Seminar at the University of Hamburg for the opportunity to work on this project.