# Multi-Object Tracking Assignment Submission

## 1. Submission Contents

This submission zip contains the following items:

- `Participants.csv`
- `requirements.txt`
- `README.md`
- `Project/final_report.ipynb`
- `Project/frame.proto`
- `Project/detection/`
- `Project/tracking/`
- `Project/tools/`
- `Project/tracking_outputs/tracking_results.gif`

The notebook `Project/final_report.ipynb` is the main report and runnable implementation for the assignment.

---

## 2. Important Note About Removed Files

The **dataset frames** and the **external SFA3D model files** were **not included in the zip submission** because of file-size limitations.

### Removed items
The following were intentionally left out:

- `Project/Dataset/data_2/frame_*.pb`
- `external/SFA3D/`
- `external/SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth`

### Why they were removed
The submission platform has a strict upload-size limit, and the dataset frames and pretrained model files are too large to package together with the report and implementation code.

---

## 3. How To Run The Notebook

To run the notebook successfully, the missing dataset and model files must be placed in the expected locations.

### 3.1 Expected folder structure



The project should be arranged like this:

```text
root/
├── external/
│   └── SFA3D/
│       ├── sfa/
│       └── checkpoints/
│           └── fpn_resnet_18/
│               └── fpn_resnet_18_epoch_300.pth
└── Project/
    ├── final_report.ipynb
    ├── frame.proto
    ├── detection/
    ├── tracking/
    ├── tools/
    ├── Dataset/
    │   └── data_2/
    │       ├── frame_0.pb
    │       ├── frame_1.pb
    │       ├── ...
    │       └── frame_n.pb
    └── tracking_outputs/
        └── tracking_results.gif

### 3.2 Required dataset location

Place the protobuf dataset frames in:

Project/Dataset/data_2/

Example:

Project/Dataset/data_2/frame_0.pb

### 3.3 Required model location

Place the external SFA3D repository in:

external/SFA3D/

and make sure the checkpoint exists at:

external/SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth