### Software requirements & container

This directory contains the different `software requirements/aspects` necessary to re-create the `computing environment` the `machine learning analyses` were run in. The original `software container`, in its `docker` form, can however also directly be retrieved via the below command.

`docker pull peerherholz/repronim_ml`

The following files can be found here:

- `environment.yml`: the `conda environment` with all `python dependencies`
- `generate_repronim_ml_image.sh`: a `shell` script that re-recreates the `docker image` via `neurodocker`
- `Dockerfile`: the `Dockerfile` produced by the `generate_repronim_ml_image.sh` script