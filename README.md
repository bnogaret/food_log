Master thesis project.

## Installation

Install:
- [Python 3](https://www.python.org/downloads/) (version: 3.5.2)
- [Caffe and PyCaffe](http://caffe.berkeleyvision.org/installation.html) (version : rc3)

    Make sure that caffe can be imported in your project with
    ```Python
    import caffe
    ```

    Then, install using the [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) (and especially the installation using the script included in caffe) these two pre-trained models:
    - [gist of the segmentation model](https://gist.github.com/jimmie33/339fd0a938ed026692267a60b44c0c58)
    - [gist of the feature model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)
- [OpenCv 3 and its Opencv python](http://opencv.org/downloads.html) (version: 3.1) with [opencv_contrib](https://github.com/opencv/opencv_contrib) (version 3.1)

    Again, make sure that opencv can be imported in your project with
    ```Python
    import cv2
    ```
- **Optional**: create a virtual environment for the project
- Clone the repository
    ```bash
    git clone https://github.com/bnogaret/food_log.git
    ```
- Install python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- Download and unzip in /data/ directory [UEC FOOD256](http://foodcam.mobi/dataset256.html) and/or [UEC FOOD100](http://foodcam.mobi/dataset100.html).

    :warning: **Be careful, I have modified added / modified some files from these archives.**

## Tests

To run the (too few) tests, execute from the root directory:
```bash
cd src/tests/
python3 -m unittest discover
```

## Documentation:

To generate the documentation, sphinx must be installed.
```bash
pip install sphinx
```

To compile the documentation, the below command must be
executed in the root directory:

```bash
sphinx-build -b html docs/ build/
```
