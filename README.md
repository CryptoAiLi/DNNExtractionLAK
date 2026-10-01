# DNNExtractionLAK

An experimental project for DNN model extraction and error detection. Signature recovery, sign recovery, and precision improvement largely follow the original algorithms with minimal changes. The main contributions are adaptations for extraction with unknown network architectures and the implementation of an error detection algorithm. The project recovers unknown hidden architectures through logits queries and evaluates missing-neuron detection using known ground-truth prefixes.

## Environment

Python 3.11 or later is required. Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

For GPU support, install a PyTorch version compatible with your CUDA environment.

## Usage

Place the model checkpoints in `assets/`, then run the following commands from the project root:

```powershell
python logits_extraction_experiment.py --model assets/pth_mnist_64x2_10.pth --initial-points 4000 --additional-points 2000
python ideal_error_detection_test.py --model assets/pth_cifar10_512x3_64_10.pth --point-count 5000
```

Both experiments save their outputs under `results/` by default.
