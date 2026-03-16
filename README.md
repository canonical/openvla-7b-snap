# openvla-7b-snap

This snap installs a hardware-optimized engine for inference with the [Openvla-7b](https://huggingface.co/openvla/openvla-7b) vla model.

Install:
```
sudo snap install openvla-7b --edge
```

Get help:
```
openvla-7b --help
```

## Resources

📚 **[Documentation](https://documentation.ubuntu.com/inference-snaps/)**, learn how to use inference snaps

## Build and install from source

Clone this repo with its submodules:
```shell
git clone https://github.com/canonical/openvla-7b-snap.git
```

Build the snap and its component:
```shell
snapcraft pack -v
```

Install the snap and its components with:
```shell
sudo snap install --dangerous *.snap && sudo snap install --dangerous *.comp
```
