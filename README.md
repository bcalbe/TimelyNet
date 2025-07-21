# TimelyNet

TimelyNet is a real-time neural architecture adaptation approach that can be integrated into existing autonomous driving pipelines to meet dynamic latency requirements.
This repository provides the inference code of TimelyNet implemented on top of the Trajectory Control Prediction (TCP) model. 

In this repository, we run latency tests and control quality tests on the TimelyNet. In latency test, we set different latency requirements from 40 ms to 90 ms, with a step of 10 ms. For each latency requirement, we predict the optimal architeture using the lookup table and invertible neural network (INN) model. We fuse the two predicted architectures as the final architecture. In control quality test, we use the predicted architecture to run the TCP model with predicted subnet architecture and evaluate the control quality of the model.



<!-- ## Features
- **Latency Collection**: Tools for measuring and analyzing latency across different neural network architectures.
- **Model Switching**: Dynamic switching between models based on latency and accuracy requirements.
- **Lookup Table**: Precomputed tables for efficient architecture selection.
- **Evaluation Metrics**: Functions for calculating accuracy, latency, and error rates. -->

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/TimelyNet.git
   cd TimelyNet
   ```
2. Create conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate TimelyNet
   ```

## Getting Started
1. Download the model with supernet encoder from the link below and place it in the `checkpoint/TCP_model/` directory.
   [Download TCP Model](https://drive.google.com/file/d/1Y0lbZFXNeIco0gIpQlxW9K8GyrrXaxZZ/view?usp=drive_link)
   
2. Run the main script to start the latency test and control quality test:
   ```bash
   python main.py
   ```

### Loading Models
Use the `load_TCP` function to load the TCP model:
```python
from TCP.config import GlobalConfig
from main import load_TCP

config = GlobalConfig()
model = load_TCP(config=config, resume_path="./checkpoint/TCP_model/res50_ta_4wp_123+10ep.ckpt")
```

### Running Tests
Run latency and control quality tests:
```python

latency_test()
control_quality_test()
```

### Desired Output:
- **latency_test**: In latency test, we measure the latency for predicted architectures under different latency requirements. The measured latency is expected to increase with higher latency requirements.

- **control_quality_test**: This function evaluates the control quality of the TCP model using the predicted subnet architecture. The control quality is expected to increase with the latency requirements, indicating that the model can adaptively switch to a more accurate architecture when the latency requirement is higher.

## Project Structure
- **TCP/**: Contains the main model and configuration files.
- **TimelyNet/**: Includes latency collection, model switching, and auxiliary tools.
- **data/**: Stores datasets and precomputed lookup tables.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

