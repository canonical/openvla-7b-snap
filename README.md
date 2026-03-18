# openvla-7b-snap

This snap provides hardware-optimized inference for the [OpenVLA 7B](https://huggingface.co/openvla/openvla-7b) vision-language-action model.

## Install from store

```shell
sudo snap install openvla-7b --edge
```

Show CLI help:

```shell
openvla-7b --help
```

## Build from source

Clone the repository:

```shell
git clone --recurse-submodules https://github.com/canonical/openvla-7b-snap.git
cd openvla-7b-snap
```

Pack the snap and components:

```shell
snapcraft pack -v
```

Install local artifacts:

```shell
sudo snap install --dangerous ./*.snap
sudo snap install --dangerous ./*.comp
```

## Engine selection

Auto-select the best engine for your hardware:

```shell
sudo openvla-7b use-engine --auto
```

List available engines:

```shell
sudo openvla-7b list-engines
```

Set a specific engine:

```shell
sudo openvla-7b use-engine generic-cpu-fastapi
sudo openvla-7b use-engine generic-cpu-xvla
sudo openvla-7b use-engine nvidia-gpu-fastapi
sudo openvla-7b use-engine nvidia-gpu-xvla
```

Check daemon status:

```shell
sudo snap services openvla-7b
```

### FastAPI multipart interface

Use this when running a `*-fastapi` engine.

```shell
curl -X POST "http://localhost:9090/act" \
  -F "text=pick up the blue block" \
  -F "image=@lego_on_table.png"
```

### XVLA JSON interface

Use this when running a `*-xvla` engine.

```shell
python3 test_vla.py \
  --url http://localhost:9090/act \
  --interface xvla \
  --image lego_on_table.png \
  --prompt "pick up the blue block"
```

## Resources

- [Inference Snaps documentation](https://documentation.ubuntu.com/inference-snaps/)
