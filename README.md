# Ather Image Classification

A binary image classifier that determines whether an image contains an Ather vehicle or some other vehicle.

## Setup

First, install the required dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Running the Image Classifier

```bash
python detect.py --image path/to/your/image.jpg
```

For example, to classify the included sample images:

```bash
python detect.py --image ather.png
```

or

```bash
python detect.py --image bike.png
```

## Output

The classifier will output one of two results:

- **"ather"** if the image is classified as an Ather vehicle
- **"other"** if it's classified as something else

## Training a New Model

If you need to train a new model, uncomment the code in the `__main__` section of `main.py`. The training will automatically use images from the `dataset` directory.
