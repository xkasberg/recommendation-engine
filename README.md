
# Plush Pipeline & Recommendation Service

It takes approximately 15 minutes to buld the plush-pipeline image the first time they are built

`docker compose up --build`

```
docker compose up --build
WARN[0000] /Users/kris/repos/ml-interview/plush/compose.yaml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
[+] Building 850.0s (24/24) FINISHED                                                                                                                           
 => [internal] load local bake definitions                                                                                                                0.0s
 => => reading from stdin 683B                                                                                                                            0.0s
 => [pipeline internal] load build definition from Dockerfile                                                                                             0.0s
 => => transferring dockerfile: 1.05kB                                                                                                                    0.0s
 => [api internal] load build definition from Dockerfile                                                                                                  0.0s
 => => transferring dockerfile: 1.17kB                                                                                                                    0.0s
 => [api internal] load metadata for docker.io/library/python:3.10-slim                                                                                   1.0s
 => [pipeline internal] load .dockerignore                                                                                                                0.0s
 => => transferring context: 2B                                                                                                                           0.0s
 => [pipeline 1/7] FROM docker.io/library/python:3.10-slim@sha256:975a1e200a16719060d391eea4ac66ee067d709cc22a32f4ca4737731eae36c0                        0.5s
 => => resolve docker.io/library/python:3.10-slim@sha256:975a1e200a16719060d391eea4ac66ee067d709cc22a32f4ca4737731eae36c0                                 0.4s
 => [pipeline internal] load build context                                                                                                                1.8s
 => => transferring context: 1.65MB                                                                                                                       1.7s
 => [api internal] load build context                                                                                                                     1.2s
 => => transferring context: 429.71kB                                                                                                                     1.1s
 => CACHED [pipeline 2/7] RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates  && rm -rf /var/lib/apt/lists/*  && curl  0.0s
 => CACHED [api stage-0 3/8] WORKDIR /app/api                                                                                                             0.0s
 => CACHED [api stage-0 4/8] COPY api/pyproject.toml /app/api/pyproject.toml                                                                              0.0s
 => CACHED [api stage-0 5/8] COPY api/uv.lock /app/api/uv.lock                                                                                            0.0s
 => [api stage-0 6/8] RUN --mount=type=cache,target=/root/.cache/uv     uv sync --frozen --no-dev                                                         7.4s
 => CACHED [pipeline 3/7] WORKDIR /app/pipeline                                                                                                           0.0s
 => CACHED [pipeline 4/7] COPY pipeline/pyproject.toml /app/pipeline/pyproject.toml                                                                       0.0s
 => CACHED [pipeline 5/7] COPY pipeline/uv.lock /app/pipeline/uv.lock                                                                                     0.0s
 => CACHED [pipeline 6/7] RUN uv sync --frozen --no-dev                                                                                                   0.0s
 => CACHED [pipeline 7/7] COPY pipeline/ /app/pipeline/                                                                                                   0.0s
 => [pipeline] exporting to image                                                                                                                       845.9s
 => => exporting layers                                                                                                                                 716.3s
 => => exporting manifest sha256:0fdf2d7ed1f665e1240e8c236c45333b9dca39794b940e58b773703b8b7e4e4a                                                         0.0s
 => => exporting config sha256:a44af1c41e202a21ebe691bbba9bd7f5b031f16df92790a61b401ef996b38223                                                           0.0s
 => => exporting attestation manifest sha256:5af007fe2bb22a9a4269aedc5b8412e6852c56334574cad3fc6091d35dc8f40a                                             0.0s
 => => exporting manifest list sha256:e18af5934fd54eef6236096706974291720ab37f8c69f41ee4baf1869fe7218c                                                    0.0s
 => => naming to docker.io/library/plush-pipeline:latest                                                                                                  0.0s
 => => unpacking to docker.io/library/plush-pipeline:latest                                                                                             129.4s
 => [api stage-0 7/8] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install torch==1.13.1                                                    112.6s
 => [api stage-0 8/8] COPY api/ /app/api/                                                                                                                 6.2s
 => [api] exporting to image                                                                                                                            371.2s
 => => exporting layers                                                                                                                                 295.3s
 => => exporting manifest sha256:e3723ee2ec933394e27da8e39bfb9dadf97e634684769d1dfbd94859a3e71ebc                                                         0.1s
 => => exporting config sha256:fe7f5ca7d7ebb8afa6327784f763b06c4e876753a13a96bb8beda89caf5fc78f                                                           0.0s
 => => exporting attestation manifest sha256:d49111d1709c00db0fbd7ae6feaf2221e461b59d5a4760903bb4cc9dd3b5f025                                             0.1s
 => => exporting manifest list sha256:d2e063d9931c34932c319ee168d0d3e0889ef5f792e138bb282f0238109e6181                                                    0.0s
 => => naming to docker.io/library/plush-api:latest                                                                                                       0.0s
 => => unpacking to docker.io/library/plush-api:latest                                                                                                   75.4s
 => [api] resolving provenance for metadata file                                                                                                          0.1s
 => [pipeline] resolving provenance for metadata file                                                                                                     0.0s
[+] Running 6/6
 ✔ pipeline                  Built                                                                                                                        0.0s 
 ✔ api                       Built                                                                                                                        0.0s 
 ✔ Network plush_default     Created                                                                                                                      0.2s 
 ✔ Volume "plush_artifacts"  Created                                                                                                                      0.0s 
 ✔ Container plush-pipeline  Created                                                                                                                      1.6s 
 ✔ Container plush-api       Created                                                                                                                      0.5s 
Attaching to plush-api, plush-pipeline
```

