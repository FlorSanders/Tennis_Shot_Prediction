# Notes

For the data processing steps, we make use of highly performat pre-trained models wherever possible. The inference code for these models are included as submodules in this repository.

However, as these original projects were developed with different goals in mind and over a wide timespan, care must be taken when trying to run them together.

## [Tenniset](https://github.com/HaydenFaulkner/Tennis)

The source of our main dataset. The code in this repository is not used as the processing tasks are different for the project.

## [Tennis Project](https://github.com/yastrebksv/TennisProject)

While alternatives like [Tennis Tracking](https://github.com/ArtLabss/tennis-tracking) offers the same functionality, they makes use of classical machine learning techniques that are quite compute intensive and cannot be parallellized on GPU.

This project offers a more modern Pytorch-based model for the tracking.  
The code can be run with the `teco` environment used for this project.

### Bug fixes

- [homography.py](./tennis-project/homography.py): `tans_kps[i]` needs to be wrapped in `np.unsqueeze()` for the code to work.
- [postprocess.py](./tennis-project/postprocess.py): In the function `line_intersection(line1, line2)`, the if-statement needs to be extended with `and not isinstance(intersection[0], Line)`.
