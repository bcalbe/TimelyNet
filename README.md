# TimelyNet

TimelyNet is a framework designed for latency-aware neural network optimization and evaluation. It integrates various components such as latency prediction, model switching, and performance evaluation to ensure efficient and accurate control in real-time systems.

## Features
- **Latency Collection**: Tools for measuring and analyzing latency across different neural network architectures.
- **Model Switching**: Dynamic switching between models based on latency and accuracy requirements.
- **Lookup Table**: Precomputed tables for efficient architecture selection.
- **Evaluation Metrics**: Functions for calculating accuracy, latency, and error rates.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/TimelyNet.git
   cd TimelyNet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Loading Models
Use the `load_TCP` function to load the TCP model:
```python
from TCP.config import GlobalConfig
from main import load_TCP

config = GlobalConfig()
model = load_TCP(config=config, resume_path="./checkpoint/TCP_result/res50_ta_4wp_123+10ep.ckpt")
```

### Running Tests
Run latency and control quality tests:
```python
from main import latency_test, control_quality_test

latency_test()
control_quality_test()
```

### Visualizing Data
Use the `show_images` function to visualize images from the dataset:
```python
from main import show_images
from torch.utils.data import DataLoader
from TCP.data import CARLA_Data

data_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)

show_images(data_loader)
```

## Project Structure
- **TCP/**: Contains the main model and configuration files.
- **TimelyNet/**: Includes latency collection, model switching, and auxiliary tools.
- **data/**: Stores datasets and precomputed lookup tables.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Inspired by real-time control systems and neural network optimization techniques.
- Special thanks to contributors and the open-source community.