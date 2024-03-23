# Notes

For the data processing steps, we make use of highly performat pre-trained models wherever possible. The inference code for these models are included as submodules in this repository.

However, as these original projects were developed with different goals in mind and over a wide timespan, care must be taken when trying to run them together.

## [Tenniset](https://github.com/HaydenFaulkner/Tennis)

The source of our main dataset. The code in this repository is not used as the processing tasks are different for the project.

## [Tennis Court Detector](https://github.com/yastrebksv/TennisCourtDetector)

While Tennis Tracking offers court detection functionality, it makes use of classical machine learning techniques that are quite compute intensive and cannot be parallellized on GPU.

This project offers a more modern Pytorch-based model for the tracking.  
The code can be run with the `teco` environment used for this project.

### Bug fixes

- [homography.py](./tennis-court-detector/homography.py): `tans_kps[i]` needs to be wrapped in `np.unsqueeze()` for the code to work.
