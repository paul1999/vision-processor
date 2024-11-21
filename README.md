# Vision Processor
A replacement for the aging (ssl-vision)[https://github.com/RoboCup-SSL/ssl-vision].
The shape based blob detector and decentralized software architecture is intended to
minimize setup time and improve detection rates in uneven illumination conditions.

## Usage

1. Compile `vision_processor` on all video data processing computers,
   setup Python on the computer intended to inspect and configure the vision.
2. Configure one `config-minimal.yml` for each camera
   (on the corresponding processing computer, skip the `geometry` section for now) and
   `geometry.yml` on the configuration computer.
3. Start `vision_processor` with correct configuration for each camera.
4. Start `python/cam_viewer.py --cameras <X>` on the configuration computer.
5. Adjust the orientation and position of the cameras.
   If blobs are overexposed (more white than colorful) set manual camera exposure and gain values (see `config.yml`). 
6. Restart `vision_processor` for the generation of a new sample image `img/sample.[X].png`
   and complete the `geometry` section of each camera config with it.
7. Restart all `vision_processor`s (to reload the config).
8. Start the `python/geom_publisher.py`.


## vision_processor setup
`vision_processor` is the camera data processing program. For each camera one instance is required.

### General dependencies
Debian/Ubuntu based distributions: `apt install g++ cmake pkg-config libyaml-cpp-dev ocl-icd-opencl-dev libeigen3-dev libopencv-dev protobuf-compiler libprotobuf-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev`

Arch based distributions: `pacman -S gcc make cmake pkgconf eigen opencv yaml-cpp opencl-clhpp`

### OpenCL runtime
A OpenCL runtime for your GPU or CPU is required.

#### Nvidia
Arch based distributions: `pacman -S opencl-nvidia`

#### Intel
Ubuntu based distributions: `apt install intel-opencl-icd intel-media-va-driver-non-free`

### Camera backends

#### Matrix Vision Bluefox, mvIMPACT
Arch user repository: `mvimpact-acquire`

#### Teledyne FLIR, Spinnaker
Arch user repository: `spinnaker-sdk`

### Compiling

    cmake -B build .
    make -C build vision_processor

### Usage
Configure `config-minimal.yml` or `config.yml` according to your needs.
Have a running geometry publisher instance in your network.

Run `build/vision_processor [Config file]`.
If you have configured `config.yml` in your working directory, you may omit the config file path.


## Python setup
Required to run any of the python scripts.

Debian/Ubuntu based distributions: `apt install cmake python3-protobuf python3-yaml python3-grpc-tools mpv`

Arch based distributions: `pacman -S make cmake python-protobuf python-yaml python-grpcio-tools mpv`

Alternative dependency installation with PIP: `pip install protobuf pyyaml grpcio-tools`

`mpv` is optional and only required for `python/cam_viewer.py`.

### python/geom_publisher.py
Publishes the field geometry for all vision_processors, teams and the game controller.
Needs generated protobuf files:

    cmake -B build -DPYTHON_ONLY .
    make -C build AUTOGENERATE

Configure one of the geometry*.yml files according to your field setup and give that file as first argument of the script.

### python/cam_viewer.py
Opens the `mpv` video player with the camera streams from the vision_processor instances.


## Troubleshooting

Look if a config option in the full `config.yml` might fix your issues.

If nothing helps, preserve the problematic scenario (Used `config.yml`, `geometry.yml` and a video of with
`ffmpeg -protocol_whitelist file,rtp,udp -i python/cam[X].sdp cam[X].mp4`) for remote analysis.
